import os
from datetime import timedelta
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class SubtitleGenerator:
    def __init__(self, output_dir="output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def format_timestamp(self, seconds, format_type="srt"):
        """
        Convert seconds to subtitle timestamp format
        """
        td = timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        secs = int(td.total_seconds() % 60)
        millisecs = int((seconds % 1) * 1000)
        
        if format_type == "srt":
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
        elif format_type == "vtt":
            return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def generate_srt(self, segments, filename="output.srt"):
        """
        Generate SRT subtitle file with speaker labels
        """
        try:
            output_path = self.output_dir / filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, segment in enumerate(segments, 1):
                    start_time = self.format_timestamp(segment['start'], "srt")
                    end_time = self.format_timestamp(segment['end'], "srt")
                    
                    # Format speaker label
                    speaker = segment['speaker'].replace('SPEAKER_', 'Speaker ')
                    text = segment['text'].strip()
                    
                    # Write SRT entry
                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"[{speaker}] {text}\n\n")
            
            logger.info(f"SRT file generated: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error generating SRT: {e}")
            raise
    
    def generate_vtt(self, segments, filename="output.vtt"):
        """
        Generate WebVTT subtitle file with speaker labels
        """
        try:
            output_path = self.output_dir / filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("WEBVTT\n\n")
                
                for i, segment in enumerate(segments, 1):
                    start_time = self.format_timestamp(segment['start'], "vtt")
                    end_time = self.format_timestamp(segment['end'], "vtt")
                    
                    # Format speaker label
                    speaker = segment['speaker'].replace('SPEAKER_', 'Speaker ')
                    text = segment['text'].strip()
                    
                    # Write VTT entry
                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"[{speaker}] {text}\n\n")
            
            logger.info(f"VTT file generated: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error generating VTT: {e}")
            raise
    
    def generate_json(self, segments, metadata=None, filename="output.json"):
        """
        Generate JSON file with detailed segment information
        """
        try:
            import json
            
            output_path = self.output_dir / filename
            
            data = {
                "metadata": metadata or {},
                "segments": segments,
                "statistics": {
                    "total_segments": len(segments),
                    "total_duration": max([s['end'] for s in segments]) if segments else 0,
                    "speakers": list(set([s['speaker'] for s in segments])),
                    "average_confidence": sum([s.get('confidence', 0) for s in segments]) / len(segments) if segments else 0
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"JSON file generated: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error generating JSON: {e}")
            raise
    
    def generate_transcript(self, segments, filename="transcript.txt"):
        """
        Generate plain text transcript with speaker labels
        """
        try:
            output_path = self.output_dir / filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("TRANSCRIPT\n")
                f.write("=" * 50 + "\n\n")
                
                current_speaker = None
                for segment in segments:
                    speaker = segment['speaker'].replace('SPEAKER_', 'Speaker ')
                    text = segment['text'].strip()
                    
                    # Add speaker label if speaker changes
                    if speaker != current_speaker:
                        f.write(f"\n{speaker}:\n")
                        current_speaker = speaker
                    
                    f.write(f"{text} ")
                
                f.write("\n\n")
                
                # Add statistics
                f.write("STATISTICS\n")
                f.write("=" * 20 + "\n")
                f.write(f"Total segments: {len(segments)}\n")
                f.write(f"Total speakers: {len(set([s['speaker'] for s in segments]))}\n")
                if segments:
                    f.write(f"Duration: {max([s['end'] for s in segments]):.1f} seconds\n")
                    f.write(f"Average confidence: {sum([s.get('confidence', 0) for s in segments]) / len(segments):.2f}\n")
            
            logger.info(f"Transcript generated: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error generating transcript: {e}")
            raise
    
    def generate_all_formats(self, segments, base_filename="output", metadata=None):
        """
        Generate all subtitle formats
        """
        try:
            results = {}
            
            # Generate SRT
            results['srt'] = self.generate_srt(segments, f"{base_filename}.srt")
            
            # Generate VTT
            results['vtt'] = self.generate_vtt(segments, f"{base_filename}.vtt")
            
            # Generate JSON
            results['json'] = self.generate_json(segments, metadata, f"{base_filename}.json")
            
            # Generate transcript
            results['transcript'] = self.generate_transcript(segments, f"{base_filename}_transcript.txt")
            
            logger.info("All subtitle formats generated successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error generating subtitle formats: {e}")
            raise
    
    def validate_segments(self, segments):
        """
        Validate segments before generating subtitles
        """
        if not segments:
            raise ValueError("No segments provided")
        
        required_fields = ['start', 'end', 'speaker', 'text']
        
        for i, segment in enumerate(segments):
            for field in required_fields:
                if field not in segment:
                    raise ValueError(f"Segment {i} missing required field: {field}")
            
            if segment['start'] >= segment['end']:
                logger.warning(f"Segment {i} has invalid timing: {segment['start']} >= {segment['end']}")
            
            if not segment['text'].strip():
                logger.warning(f"Segment {i} has empty text")
        
        # Sort segments by start time
        segments.sort(key=lambda x: x['start'])
        
        return segments