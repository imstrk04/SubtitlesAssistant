import os
import yt_dlp
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoDownloader:
    def __init__(self, output_dir="temp"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def download_video(self, url, max_duration=600):
        """
        Download video from URL with audio extraction
        Args:
            url: Video URL (YouTube, etc.)
            max_duration: Maximum duration in seconds (default 10 mins)
        Returns:
            Path to downloaded audio file
        """
        try:
            # First, get video info without downloading
            info_opts = {
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(info_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                title = info.get('title', 'video')
                duration = info.get('duration', 0)
                
                logger.info(f"Video: {title} ({duration}s)")
                
                if duration > max_duration:
                    raise ValueError(f"Video too long: {duration}s > {max_duration}s")
            
            # Clean up any existing files first
            for file in self.output_dir.iterdir():
                if file.is_file() and file.suffix in ['.wav', '.mp3', '.m4a']:
                    file.unlink()
            
            # Now download with proper options - FIXED template
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': str(self.output_dir / 'audio.%(ext)s'),  # This should work
                'ffmpeg_location': r'C:\\ffmpeg',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'keepvideo': False,
                'restrictfilenames': True,  # Force simple filenames
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
                
                logger.info("🔍 Looking for downloaded audio files...")
                
                # Look for the expected file first
                expected_file = self.output_dir / 'audio.wav'
                if expected_file.exists():
                    logger.info(f"✅ Found expected file: {expected_file}")
                    return str(expected_file)
                
                # Fallback: look for any audio files
                all_files = list(self.output_dir.iterdir())
                audio_files = [f for f in all_files if f.suffix.lower() in ['.wav', '.mp3', '.m4a', '.webm', '.ogg']]
                
                logger.info(f"Found {len(all_files)} total files, {len(audio_files)} audio files")
                logger.info(f"All files: {[f.name for f in all_files]}")
                logger.info(f"Audio files: {[f.name for f in audio_files]}")
                
                if audio_files:
                    # Get the newest file
                    latest_file = max(audio_files, key=lambda f: f.stat().st_mtime)
                    logger.info(f"✅ Selected audio file: {latest_file.name}")
                    logger.info(f"📏 File size: {latest_file.stat().st_size} bytes")
                    return str(latest_file)
                else:
                    raise FileNotFoundError(f"No audio files found. Directory contents: {[f.name for f in all_files]}")
                    
        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            # Show what files exist for debugging
            try:
                existing_files = list(self.output_dir.iterdir())
                logger.info(f"Files in directory after error: {[f.name for f in existing_files]}")
            except:
                pass
            raise
    
    def process_local_video(self, video_path):
        """
        Extract audio from local video file using ffmpeg
        Args:
            video_path: Path to local video file
        Returns:
            Path to extracted audio file
        """
        try:
            import subprocess
            
            video_path = Path(video_path)
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            output_path = self.output_dir / f"{video_path.stem}.wav"
            
            logger.info(f"Extracting audio from: {video_path}")
            
            # Use ffmpeg directly via subprocess
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', '44100',  # Sample rate
                '-ac', '2',  # Stereo
                '-y',  # Overwrite output file
                str(output_path)
            ]
            
            # Run ffmpeg command
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=False
            )
            
            if result.returncode != 0:
                # Try alternative approach with ffmpeg-python if available
                try:
                    import ffmpeg
                    
                    logger.info("Trying with ffmpeg-python library...")
                    
                    stream = ffmpeg.input(str(video_path))
                    audio = stream.audio
                    out = ffmpeg.output(
                        audio, 
                        str(output_path),
                        acodec='pcm_s16le',
                        ar=44100,
                        ac=2
                    )
                    ffmpeg.run(out, overwrite_output=True, quiet=True)
                    
                except ImportError:
                    raise RuntimeError(f"FFmpeg failed: {result.stderr}")
                except Exception as e:
                    raise RuntimeError(f"FFmpeg-python failed: {e}")
            
            if not output_path.exists():
                raise RuntimeError("Audio extraction failed - no output file created")
            
            logger.info(f"Audio extracted to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error processing local video: {e}")
            raise
    
    def cleanup(self):
        """Clean up all files in the output directory"""
        try:
            for file in self.output_dir.iterdir():
                if file.is_file():
                    file.unlink()
                    logger.info(f"Cleaned up: {file}")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    def list_downloaded_files(self):
        """List all files in the output directory"""
        files = list(self.output_dir.iterdir())
        logger.info(f"Files in {self.output_dir}:")
        for file in files:
            logger.info(f"  - {file.name} ({file.stat().st_size} bytes)")
        return files

# Test script
if __name__ == "__main__":
    # Test the downloader
    downloader = VideoDownloader()
    
    try:
        # Test with the problematic video
        video_url = "https://youtu.be/zYJKq17GpEc?si=0apoU-vLWrmJfFox"
        print(f"🎬 Testing download: {video_url}")
        
        audio_path = downloader.download_video(video_url, max_duration=300)
        print(f"✅ SUCCESS! Audio saved to: {audio_path}")
        
        # Verify file exists
        if os.path.exists(audio_path):
            print(f"📏 File size: {os.path.getsize(audio_path)} bytes")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        # Show debug info
        downloader.list_downloaded_files()