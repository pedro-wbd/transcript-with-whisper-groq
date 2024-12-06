# Groq Audio Transcription Tool

## Overview
A Python script for transcribing large audio files using the Groq API, with support for:
- Handling files larger than 25MB
- Audio downsampling
- Chunk-based transcription
- Flexible language and prompt options

## Prerequisites
- Python 3.8+
- FFmpeg installed
- Groq API Key

## Setup

### 1. Clone the Repository
```bash
git clone https://your-repo-url.git
cd audio-transcription-tool
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

### Command Line
```bash
python transcribe.py input_audio.mp3 output_transcription.txt --language en
```

### Optional Parameters
- `--language`: Specify audio language (ISO 639-1 code)
- `--prompt`: Add context for transcription

## AWS Lambda Deployment
1. Zip the project including:
   - `transcribe.py`
   - `requirements.txt`
   - Installed dependencies
2. Set `GROQ_API_KEY` as an environment variable in Lambda

## Troubleshooting
- Ensure FFmpeg is installed
- Check API key permissions
- Verify audio file format

## License
[Your License Here]