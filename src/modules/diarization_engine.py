import torch
import whisper
import numpy as np
from pyannote.audio import Pipeline
from pyannote.core import Segment, Annotation
import logging
from pathlib import Path
import warnings
import gc
import psutil
import os
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("HUGGINGFACE_AUTH_TOKEN")

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class AdvancedDiarizationEngine:
    def __init__(self, 
                 whisper_model="small",  # Default to smaller model
                 diarization_model="pyannote/speaker-diarization-3.0",
                 device='cpu',
                 force_cpu=False,
                 low_memory_mode=True):
        """
        Advanced diarization engine combining Whisper + Pyannote
        Optimized for memory efficiency
        """
        
        # Memory management
        self.low_memory_mode = low_memory_mode
        if self.low_memory_mode:
            self._cleanup_gpu_memory()
        
        # Device selection with memory considerations
        self.device = self._select_optimal_device(device, force_cpu)
        logger.info(f"Using device: {self.device}")
        
        # Model size adjustment based on available memory
        whisper_model = self._adjust_model_size(whisper_model)
        
        # Initialize Whisper for transcription
        logger.info(f"Loading Whisper model: {whisper_model}")
        try:
            self.whisper_model = whisper.load_model(
                whisper_model, 
                device=self.device,
                in_memory=False  # Don't load into memory if not needed
            )
            logger.info("Whisper model loaded successfully")
        except torch.cuda.OutOfMemoryError as e:
            logger.warning(f"GPU OOM loading Whisper, falling back to CPU: {e}")
            self.device = "cpu"
            self.whisper_model = whisper.load_model(whisper_model, device="cpu")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            # Try with smallest model as fallback
            logger.info("Attempting fallback to 'tiny' model")
            self.whisper_model = whisper.load_model("tiny", device="cpu")
        
        # Initialize Pyannote for diarization (with memory optimization)
        logger.info("Loading Pyannote diarization pipeline")
        try:
            self.diarization_pipeline = Pipeline.from_pretrained(
                diarization_model,
                use_auth_token= token
            )
            
            # Only move to GPU if we have enough memory and it's not CPU-only
            if self.device != "cpu" and torch.cuda.is_available():
                try:
                    self.diarization_pipeline = self.diarization_pipeline.to(torch.device(self.device))
                    logger.info("Diarization pipeline moved to GPU")
                except torch.cuda.OutOfMemoryError:
                    logger.warning("Not enough GPU memory for diarization pipeline, using CPU")
                    # Pipeline stays on CPU by default
            
        except Exception as e:
            logger.warning(f"Could not load {diarization_model}: {e}")
            logger.info("Using fallback diarization method")
            self.diarization_pipeline = None
    
    def _cleanup_gpu_memory(self):
        """Clean up GPU memory before initialization"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            # Try to clear any existing CUDA context
            try:
                torch.cuda.synchronize()
            except:
                pass
    
    def _get_available_memory(self):
        """Get available system and GPU memory"""
        # System RAM
        ram = psutil.virtual_memory()
        available_ram_gb = ram.available / (1024**3)
        
        # GPU memory
        gpu_available = 0
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_allocated = torch.cuda.memory_allocated(0)
                gpu_cached = torch.cuda.memory_reserved(0)
                gpu_available = (gpu_memory - gpu_allocated - gpu_cached) / (1024**3)
            except:
                gpu_available = 0
        
        return available_ram_gb, gpu_available
    
    def _select_optimal_device(self, device, force_cpu):
        """Select optimal device based on memory availability"""
        if force_cpu:
            return "cpu"
        
        if device:
            return device
        
        if not torch.cuda.is_available():
            return "cpu"
        
        ram_gb, gpu_gb = self._get_available_memory()
        logger.info(f"Available memory - RAM: {ram_gb:.1f}GB, GPU: {gpu_gb:.1f}GB")
        
        # Need at least 2GB GPU memory for medium model, 1GB for small
        if gpu_gb < 1.0:
            logger.warning("Insufficient GPU memory, using CPU")
            return "cpu"
        elif gpu_gb < 2.0:
            logger.warning("Limited GPU memory, consider using 'small' model")
        
        return "cuda"
    
    def _adjust_model_size(self, requested_model):
        """Adjust model size based on available memory"""
        ram_gb, gpu_gb = self._get_available_memory()
        
        # Model memory requirements (approximate)
        model_requirements = {
            "large": {"gpu": 3.0, "ram": 4.0},
            "medium": {"gpu": 2.0, "ram": 3.0},
            "small": {"gpu": 1.0, "ram": 2.0},
            "base": {"gpu": 0.5, "ram": 1.0},
            "tiny": {"gpu": 0.3, "ram": 0.5}
        }
        
        memory_gb = gpu_gb if self.device != "cpu" else ram_gb
        
        # Find largest model that fits in memory
        model_priority = ["large", "medium", "small", "base", "tiny"]
        
        for model in model_priority:
            required = model_requirements[model]
            required_memory = required["gpu"] if self.device != "cpu" else required["ram"]
            
            if memory_gb >= required_memory:
                if model != requested_model:
                    logger.warning(f"Adjusted model from '{requested_model}' to '{model}' due to memory constraints")
                return model
        
        # Fallback to tiny if nothing else fits
        logger.warning("Using 'tiny' model as fallback due to severe memory constraints")
        return "tiny"
    
    def transcribe_audio(self, audio_path):
        """
        Transcribe audio using Whisper with memory optimization
        """
        try:
            logger.info("Transcribing audio with Whisper...")
            
            if self.low_memory_mode:
                self._cleanup_gpu_memory()
            
            # Use smaller chunks for large files if in low memory mode
            transcribe_options = {
                "word_timestamps": True,
                "verbose": False
            }
            
            # Add memory optimization options
            if self.low_memory_mode:
                transcribe_options.update({
                    "fp16": self.device != "cpu",  # Use FP16 on GPU to save memory
                    "condition_on_previous_text": False,  # Reduce memory usage
                })
            
            result = self.whisper_model.transcribe(audio_path, **transcribe_options)
            
            # Extract word-level information
            words = []
            for segment in result["segments"]:
                if "words" in segment:
                    for word in segment["words"]:
                        words.append({
                            'start': word.get('start', segment['start']),
                            'end': word.get('end', segment['end']),
                            'word': word.get('word', '').strip(),
                            'confidence': word.get('probability', 0.0)
                        })
                else:
                    # Fallback if word timestamps not available
                    words.append({
                        'start': segment['start'],
                        'end': segment['end'],
                        'word': segment['text'].strip(),
                        'confidence': 0.8
                    })
            
            logger.info(f"Transcribed {len(words)} words")
            
            # Clean up memory after transcription
            if self.low_memory_mode:
                self._cleanup_gpu_memory()
            
            return {
                'text': result['text'],
                'words': words,
                'segments': result['segments']
            }
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU OOM during transcription: {e}")
            logger.info("Retrying transcription on CPU...")
            
            # Move model to CPU and retry
            self.whisper_model = self.whisper_model.cpu()
            self.device = "cpu"
            self._cleanup_gpu_memory()
            
            # Retry with CPU
            result = self.whisper_model.transcribe(
                audio_path,
                word_timestamps=True,
                verbose=False
            )
            
            words = []
            for segment in result["segments"]:
                if "words" in segment:
                    for word in segment["words"]:
                        words.append({
                            'start': word.get('start', segment['start']),
                            'end': word.get('end', segment['end']),
                            'word': word.get('word', '').strip(),
                            'confidence': word.get('probability', 0.0)
                        })
            
            return {
                'text': result['text'],
                'words': words,
                'segments': result['segments']
            }
            
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            raise
    
    def perform_diarization(self, audio_path):
        """
        Perform speaker diarization using Pyannote with memory optimization
        """
        try:
            if self.diarization_pipeline is None:
                logger.warning("Using fallback diarization method")
                return self._fallback_diarization(audio_path)
            
            logger.info("Performing speaker diarization...")
            
            if self.low_memory_mode:
                self._cleanup_gpu_memory()
            
            # Run diarization
            diarization = self.diarization_pipeline(audio_path)
            
            # Convert to our format
            speaker_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker,
                    'duration': turn.end - turn.start
                })
            
            # Sort by start time
            speaker_segments.sort(key=lambda x: x['start'])
            
            logger.info(f"Found {len(set([s['speaker'] for s in speaker_segments]))} speakers")
            logger.info(f"Generated {len(speaker_segments)} speaker segments")
            
            # Clean up memory after diarization
            if self.low_memory_mode:
                self._cleanup_gpu_memory()
            
            return speaker_segments
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU OOM during diarization: {e}")
            logger.info("Retrying diarization on CPU...")
            
            # Move pipeline to CPU if possible
            try:
                self.diarization_pipeline = self.diarization_pipeline.cpu()
                self._cleanup_gpu_memory()
                
                diarization = self.diarization_pipeline(audio_path)
                speaker_segments = []
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    speaker_segments.append({
                        'start': turn.start,
                        'end': turn.end,
                        'speaker': speaker,
                        'duration': turn.end - turn.start
                    })
                speaker_segments.sort(key=lambda x: x['start'])
                return speaker_segments
                
            except Exception as fallback_error:
                logger.warning(f"CPU fallback failed: {fallback_error}")
                return self._fallback_diarization(audio_path)
            
        except Exception as e:
            logger.error(f"Error in diarization: {e}")
            return self._fallback_diarization(audio_path)
    
    def _fallback_diarization(self, audio_path):
        """
        Fallback diarization using energy-based speaker change detection
        """
        logger.info("Using fallback energy-based diarization")
        
        try:
            import librosa
        except ImportError:
            logger.error("librosa not installed, cannot perform fallback diarization")
            # Return a single speaker segment
            import wave
            try:
                with wave.open(str(audio_path), 'rb') as wav_file:
                    duration = wav_file.getnframes() / wav_file.getframerate()
            except:
                duration = 60.0  # Default duration
            
            return [{
                'start': 0,
                'end': duration,
                'speaker': 'SPEAKER_01',
                'duration': duration
            }]
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Simple energy-based segmentation
        hop_length = 512
        frame_length = 2048
        
        # Compute energy
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Detect speaker changes (simplified)
        threshold = np.percentile(energy, 30)
        speaker_changes = []
        
        current_speaker = 1
        segment_start = 0
        
        for i, e in enumerate(energy):
            time = i * hop_length / sr
            if e < threshold and time - segment_start > 2.0:  # Minimum 2s segments
                speaker_changes.append({
                    'start': segment_start,
                    'end': time,
                    'speaker': f'SPEAKER_{current_speaker:02d}',
                    'duration': time - segment_start
                })
                segment_start = time
                current_speaker = 2 if current_speaker == 1 else 1
        
        # Add final segment
        final_time = len(audio) / sr
        speaker_changes.append({
            'start': segment_start,
            'end': final_time,
            'speaker': f'SPEAKER_{current_speaker:02d}',
            'duration': final_time - segment_start
        })
        
        return speaker_changes
    
    def align_transcription_with_speakers(self, transcription, speaker_segments):
        """
        Align Whisper transcription with speaker diarization
        """
        logger.info("Aligning transcription with speaker segments...")
        
        aligned_segments = []
        words = transcription['words']
        
        if not words:
            return aligned_segments
        
        for speaker_seg in speaker_segments:
            # Find words that overlap with this speaker segment
            segment_words = []
            
            for word in words:
                # Check if word overlaps with speaker segment
                word_start = word['start']
                word_end = word['end']
                
                # Calculate overlap
                overlap_start = max(word_start, speaker_seg['start'])
                overlap_end = min(word_end, speaker_seg['end'])
                
                if overlap_end > overlap_start:
                    # There's overlap
                    overlap_ratio = (overlap_end - overlap_start) / (word_end - word_start)
                    
                    if overlap_ratio > 0.5:  # More than 50% overlap
                        segment_words.append(word)
            
            if segment_words:
                # Create aligned segment
                text = ' '.join([w['word'] for w in segment_words])
                avg_confidence = sum([w['confidence'] for w in segment_words]) / len(segment_words)
                
                aligned_segments.append({
                    'start': speaker_seg['start'],
                    'end': speaker_seg['end'],
                    'speaker': speaker_seg['speaker'],
                    'text': text.strip(),
                    'confidence': avg_confidence,
                    'word_count': len(segment_words)
                })
        
        # Post-process to ensure no overlaps and fill gaps
        aligned_segments = self._post_process_segments(aligned_segments)
        
        logger.info(f"Created {len(aligned_segments)} aligned segments")
        return aligned_segments
    
    def _post_process_segments(self, segments):
        """
        Post-process segments to ensure quality
        """
        if not segments:
            return segments
        
        # Sort by start time
        segments.sort(key=lambda x: x['start'])
        
        # Remove very short segments (< 0.5s)
        segments = [s for s in segments if s['end'] - s['start'] >= 0.5]
        
        # Merge consecutive segments from same speaker
        merged_segments = []
        current_segment = segments[0].copy()
        
        for next_segment in segments[1:]:
            # If same speaker and close in time (< 1s gap)
            if (current_segment['speaker'] == next_segment['speaker'] and
                next_segment['start'] - current_segment['end'] < 1.0):
                
                # Merge segments
                current_segment['end'] = next_segment['end']
                current_segment['text'] += ' ' + next_segment['text']
                current_segment['confidence'] = (
                    current_segment['confidence'] + next_segment['confidence']
                ) / 2
                current_segment['word_count'] += next_segment['word_count']
            else:
                merged_segments.append(current_segment)
                current_segment = next_segment.copy()
        
        merged_segments.append(current_segment)
        
        return merged_segments
    
    def process_audio(self, audio_path):
        """
        Main processing pipeline with memory management
        """
        logger.info(f"Processing audio: {audio_path}")
        
        try:
            # Step 1: Transcribe
            logger.info("Step 1: Transcription")
            transcription = self.transcribe_audio(audio_path)
            
            # Step 2: Diarize
            logger.info("Step 2: Diarization")
            speaker_segments = self.perform_diarization(audio_path)
            
            # Step 3: Align
            logger.info("Step 3: Alignment")
            aligned_segments = self.align_transcription_with_speakers(
                transcription, speaker_segments
            )
            
            # Final memory cleanup
            if self.low_memory_mode:
                self._cleanup_gpu_memory()
            
            return {
                'transcription': transcription,
                'speaker_segments': speaker_segments,
                'aligned_segments': aligned_segments,
                'metadata': {
                    'total_speakers': len(set([s['speaker'] for s in speaker_segments])),
                    'total_segments': len(aligned_segments),
                    'total_duration': max([s['end'] for s in aligned_segments]) if aligned_segments else 0,
                    'model_used': f"whisper-{getattr(getattr(self.whisper_model, 'dims', None), 'n_mels', 'unknown')}",
                    'device_used': self.device
                }
            }
            
        except Exception as e:
            logger.error(f"Error in process_audio: {e}")
            # Final cleanup on error
            if self.low_memory_mode:
                self._cleanup_gpu_memory()
            raise