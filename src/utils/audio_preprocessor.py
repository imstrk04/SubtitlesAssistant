import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class AudioPreprocessor:
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr
    
    def preprocess_audio(self, audio_path, output_dir="temp"):
        """
        Preprocess audio for optimal diarization
        Args:
            audio_path: Path to input audio file
            output_dir: Directory to save processed audio
        Returns:
            Path to processed audio file
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)
            logger.info(f"Loaded audio: {len(audio)/sr:.2f}s at {sr}Hz")
            
            # Resample to target sample rate
            if sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
                logger.info(f"Resampled to {self.target_sr}Hz")
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            # Remove silence (optional but can help)
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
            
            # Save processed audio
            output_path = Path(output_dir) / f"processed_{Path(audio_path).stem}.wav"
            sf.write(output_path, audio_trimmed, self.target_sr)
            
            logger.info(f"Processed audio saved: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            raise
    
    def get_audio_info(self, audio_path):
        """Get basic audio information"""
        try:
            audio, sr = librosa.load(audio_path, sr=None)
            duration = len(audio) / sr
            
            return {
                'duration': duration,
                'sample_rate': sr,
                'channels': 1 if audio.ndim == 1 else audio.shape[0],
                'samples': len(audio)
            }
        except Exception as e:
            logger.error(f"Error getting audio info: {e}")
            return None