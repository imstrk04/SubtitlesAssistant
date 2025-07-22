# Subtitle Assistant: Speaker Diarization and Quality Evaluation Pipeline

This project was built as part of an AI Engineer Intern assignment. It implements an agentic pipeline to generate high-quality speaker-labeled subtitles and evaluate the diarization quality automatically.

## Overview

The pipeline focuses on two main goals:

1. **Speaker Diarization and Transcription**

   * Generate sentence-level subtitles from a multi-speaker video/audio.
   * Accurately identify speaker turns and label them consistently.
   * Output the result in a standard subtitle format (SRT).

2. **Quality Check AI Agent**

   * Assess the generated subtitles for diarization quality.
   * Provide both a confidence score and qualitative feedback on speaker turn accuracy, label consistency, and overall clarity.

## Features

* Supports multi-speaker videos (5 minutes or longer).
* Outputs subtitle files in `.srt` format with speaker labels.
* Modular design to allow easy replacement of diarization or transcription models.
* Automated quality feedback using a local or open-source LLM.

## Project Structure

```
.
├── agentic_pipeline_sadakoparamakrishnanthothathiri.ipynb
├── src/
│   ├── modules/
│   │   ├── diarization_engine.py         # Speaker diarization logic
│   │   ├── subtitle_generator.py       # Audio transcription logic
│   │   ├── quality_agent.py              # Quality check agent logic
│   ├── utils/
│   │   ├── audio_processor.py
│   │   ├── video_downloader.py
├── output/
│   └── subtitles_quality_report.json
│   └── subtitles.srt                        # Final subtitle output
├── temp/
    └── audio.wav
    └── processed_audio.wav
```

## Methodology

1. **Audio Preprocessing**

   * Extracts audio from video using FFmpeg.
   * Applies noise reduction and normalization.

2. **Diarization**

   * Utilizes an open-source diarization model (`pyannote-audio`) to detect speaker turns.
   * Maps speaker embeddings to consistent labels (Speaker A, B, etc.).

3. **Transcription**

   * Uses a sentence-level transcription model (Whisper).
   * Aligns transcript with diarized speaker segments.

4. **Subtitle Formatting**

   * Converts aligned transcripts into SRT format.
   * Speaker labels are inserted per subtitle block.

5. **Quality Check Agent**

   * Uses a local LLM (Ollama or similar) to evaluate:

     * Speaker turn accuracy
     * Consistency of speaker labels
     * Clarity of subtitles
   * Returns a confidence score and qualitative feedback for each segment or chunk.

## Example Output (SRT)

```
1
00:00:00,000 --> 00:00:02,240
[Speaker 01] Can I ask a question? Of course you can.

2
00:00:02,240 --> 00:00:04,256
[Speaker 02] Can

3
00:00:04,256 --> 00:00:06,272
[Speaker 01] you get me out of this place?
```

## Getting Started

1. Clone the repository
2. Open the `agentic_pipeline_sadakoparamakrishnanthothathiri.ipynb` notebook in Google Colab
3. Follow the cells sequentially to:

   * Upload your video
   * Run diarization and transcription
   * Generate and download subtitle file
   * View the quality report

## Requirements

This notebook is built for execution in Google Colab and will auto-install all necessary packages. It supports:

* Whisper
* pyannote-audio
* Ollama (for local LLM-based QC)
* FFmpeg

## Limitations

* May require GPU for faster inference
* Quality agent performance depends on the base LLM's ability to interpret subtitle semantics
* Speaker label consistency is maintained across chunks heuristically; future improvements can involve clustering across longer durations

## Future Improvements

* Global speaker embedding clustering to maintain speaker labels across full videos
* Integration with multilingual ASR models
* Visual subtitle preview alongside waveform or video player
* Export to VTT and JSON formats

## License

This project is intended for evaluation and educational purposes. Please review the terms of any third-party models or datasets used.
