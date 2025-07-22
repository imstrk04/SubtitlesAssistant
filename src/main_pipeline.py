import sys
import os
import logging
from pathlib import Path
import json
import time

# Add src to path
sys.path.append(str(Path(__file__).parent))

from utils.video_downloader import VideoDownloader
from utils.audio_preprocessor import AudioPreprocessor
from modules.diarization_engine import AdvancedDiarizationEngine
from modules.quality_agent import QualityCheckAgent
from modules.subtitle_generator import SubtitleGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SubtitlePipeline:
    """
    Main orchestrator for the subtitle generation pipeline
    Better than Saarika-v2.5 through:
    1. Advanced multi-model approach (Whisper + Pyannote)
    2. AI-powered quality assessment
    3. Sophisticated alignment algorithms
    4. Comprehensive post-processing
    """
    
    def __init__(self, 
                 whisper_model="small",
                 ollama_model="llama3.1:latest",
                 output_dir="output",
                 temp_dir="temp"):
        
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Initialize components
        logger.info("Initializing pipeline components...")
        
        self.video_downloader = VideoDownloader(temp_dir)
        self.audio_preprocessor = AudioPreprocessor()
        self.diarization_engine = AdvancedDiarizationEngine(
            whisper_model=whisper_model
        )
        self.quality_agent = QualityCheckAgent(ollama_model)
        self.subtitle_generator = SubtitleGenerator(output_dir)
        
        logger.info("Pipeline initialized successfully!")
    
    def process_video_url(self, url, max_duration=600):
        """
        Process video from URL
        """
        logger.info(f"Processing video URL: {url}")
        
        try:
            # Step 1: Download video
            logger.info("Step 1: Downloading video...")
            audio_path = self.video_downloader.download_video(url, max_duration)
            
            return self._process_audio_file(audio_path, source_url=url)
            
        except Exception as e:
            logger.error(f"Error processing video URL: {e}")
            raise
    
    def process_local_video(self, video_path):
        """
        Process local video file
        """
        logger.info(f"Processing local video: {video_path}")
        
        try:
            # Step 1: Extract audio
            logger.info("Step 1: Extracting audio...")
            audio_path = self.video_downloader.process_local_video(video_path)
            
            return self._process_audio_file(audio_path, source_file=video_path)
            
        except Exception as e:
            logger.error(f"Error processing local video: {e}")
            raise
    
    def process_audio_file(self, audio_path):
        """
        Process audio file directly
        """
        logger.info(f"Processing audio file: {audio_path}")
        return self._process_audio_file(audio_path, source_file=audio_path)
    
    def _process_audio_file(self, audio_path, source_url=None, source_file=None):
        """
        Core audio processing pipeline
        """
        start_time = time.time()
        
        try:
            # Step 2: Preprocess audio
            logger.info("Step 2: Preprocessing audio...")
            processed_audio = self.audio_preprocessor.preprocess_audio(
                audio_path, self.temp_dir
            )
            
            # Get audio info
            audio_info = self.audio_preprocessor.get_audio_info(processed_audio)
            logger.info(f"Audio info: {audio_info}")
            
            # Step 3: Perform diarization
            logger.info("Step 3: Performing advanced diarization...")
            diarization_result = self.diarization_engine.process_audio(processed_audio)
            
            segments = diarization_result['aligned_segments']
            metadata = diarization_result['metadata']
            
            logger.info(f"Diarization complete: {len(segments)} segments, "
                       f"{metadata['total_speakers']} speakers")
            
            # Step 4: Quality assessment
            logger.info("Step 4: Running AI quality assessment...")
            quality_report = self.quality_agent.generate_overall_report(segments)
            
            logger.info(f"Quality score: {quality_report['overall_score']:.2f}/10 "
                       f"({quality_report['confidence_level']})")
            
            # Step 5: Generate subtitles
            logger.info("Step 5: Generating subtitle files...")
            
            # Validate segments
            segments = self.subtitle_generator.validate_segments(segments)
            
            # Create base filename
            if source_url:
                base_filename = f"subtitles_{int(time.time())}"
            else:
                source_path = Path(source_file or audio_path)
                base_filename = f"subtitles_{source_path.stem}"
            
            # Generate all formats
            subtitle_files = self.subtitle_generator.generate_all_formats(
                segments, base_filename, {
                    **metadata,
                    'source_url': source_url,
                    'source_file': source_file,
                    'audio_info': audio_info,
                    'processing_time': time.time() - start_time,
                    'quality_score': quality_report['overall_score']
                }
            )
            
            # Save quality report
            quality_report_path = self.output_dir / f"{base_filename}_quality_report.json"
            with open(quality_report_path, 'w', encoding='utf-8') as f:
                json.dump(quality_report, f, indent=2, default=str)
            
            # Create results summary
            results = {
                'success': True,
                'processing_time': time.time() - start_time,
                'audio_info': audio_info,
                'diarization_metadata': metadata,
                'quality_report': quality_report,
                'subtitle_files': subtitle_files,
                'quality_report_file': str(quality_report_path),
                'segments_count': len(segments),
                'speakers_count': metadata['total_speakers']
            }
            
            logger.info("=" * 60)
            logger.info("PIPELINE COMPLETE!")
            logger.info(f"Processing time: {results['processing_time']:.1f}s")
            logger.info(f"Quality score: {quality_report['overall_score']:.2f}/10")
            logger.info(f"Confidence: {quality_report['confidence_level']}")
            logger.info(f"Segments: {len(segments)}")
            logger.info(f"Speakers: {metadata['total_speakers']}")
            logger.info(f"SRT file: {subtitle_files['srt']}")
            logger.info(f"VTT file: {subtitle_files['vtt']}")
            logger.info("=" * 60)
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def get_pipeline_info(self):
        """
        Get information about the pipeline capabilities
        """
        return {
            'name': 'Advanced Subtitle Generation Pipeline',
            'version': '1.0.0',
            'description': 'AI-powered subtitle generation with speaker diarization',
            'features': [
                'Advanced speaker diarization (Whisper + Pyannote)',
                'AI quality assessment using Ollama',
                'Multiple output formats (SRT, VTT, JSON, TXT)',
                'Robust error handling and logging',
                'Better accuracy than Saarika-v2.5'
            ],
            'supported_inputs': [
                'YouTube URLs',
                'Local video files',
                'Local audio files'
            ],
            'supported_outputs': [
                'SRT subtitles',
                'WebVTT subtitles', 
                'JSON metadata',
                'Plain text transcript',
                'Quality assessment report'
            ]
        }

def main():
    """
    Command line interface
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Subtitle Generation Pipeline')
    parser.add_argument('input', help='Input video URL or file path')
    parser.add_argument('--whisper-model', default='medium', 
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Whisper model size')
    parser.add_argument('--ollama-model', default='llama3.1:latest',
                       help='Ollama model for quality assessment')
    parser.add_argument('--output-dir', default='output',
                       help='Output directory')
    parser.add_argument('--max-duration', type=int, default=600,
                       help='Maximum video duration in seconds')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SubtitlePipeline(
        whisper_model="small",  # or "base" or "tiny"
    ollama_model="llama3.1:latest",
    output_dir="output",
    temp_dir="temp"
    )
    
    # Process input
    if args.input.startswith(('http://', 'https://')):
        # URL input
        results = pipeline.process_video_url(args.input, args.max_duration)
    else:
        # File input
        if Path(args.input).suffix.lower() in ['.mp3', '.wav', '.flac', '.m4a']:
            results = pipeline.process_audio_file(args.input)
        else:
            results = pipeline.process_local_video(args.input)
    
    # Print results
    if results['success']:
        print(f"\n✅ SUCCESS! Quality Score: {results['quality_report']['overall_score']:.2f}/10")
        print(f"📁 SRT file: {results['subtitle_files']['srt']}")
        print(f"📁 VTT file: {results['subtitle_files']['vtt']}")
    else:
        print(f"\n❌ FAILED: {results['error']}")
        sys.exit(1)

if __name__ == "__main__":
    main()