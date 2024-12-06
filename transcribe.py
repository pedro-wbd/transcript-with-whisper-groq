import os
import subprocess
import argparse
from typing import List, Optional
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize Groq client with environment variable
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class AudioFilePreprocessor:
    """Handles preprocessing of large audio files."""
    
    @staticmethod
    def downsample_audio(input_file: str, output_file: str) -> None:
        """
        Downsample audio file to 16kHz mono using ffmpeg.
        
        Args:
            input_file (str): Path to the input audio file
            output_file (str): Path to save the downsampled audio file
        """
        try:
            # Downsample audio to 16kHz mono
            subprocess.run([
                'ffmpeg', 
                '-i', input_file, 
                '-ar', '16000', 
                '-ac', '1', 
                '-map', '0:a:', 
                output_file
            ], check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Audio downsampling failed: {e}")

    @staticmethod
    def split_audio_file(input_file: str, max_size_mb: int = 25) -> List[str]:
        """
        Split large audio files into chunks no larger than max_size_mb.
        
        Args:
            input_file (str): Path to the input audio file
            max_size_mb (int): Maximum file size in megabytes
        
        Returns:
            List[str]: Paths to the generated audio file chunks
        """
        file_size_bytes = os.path.getsize(input_file)
        max_size_bytes = max_size_mb * 1024 * 1024
        
        # If file is small enough, return the original file
        if file_size_bytes <= max_size_bytes:
            return [input_file]
        
        # Prepare chunk files
        base_name = os.path.splitext(input_file)[0]
        chunk_files = []
        
        # Use ffmpeg to split the file
        try:
            subprocess.run([
                'ffmpeg', 
                '-i', input_file, 
                '-f', 'segment', 
                '-segment_time', '300',  # 5-minute segments 
                '-segment_size', str(max_size_bytes),
                f'{base_name}_chunk_%03d.mp3'
            ], check=True)
            
            # Find all chunk files
            chunk_files = sorted([
                f for f in os.listdir(os.path.dirname(input_file) or '.')
                if f.startswith(os.path.basename(base_name) + '_chunk_') 
                and f.endswith('.mp3')
            ])
            
            return [os.path.join(os.path.dirname(input_file) or '.', f) for f in chunk_files]
        
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Audio file splitting failed: {e}")

class GroqTranscriber:
    """Handles transcription using Groq API."""
    
    def __init__(self, model: str = "whisper-large-v3-turbo"):
        """
        Initialize Groq client for transcription.
        
        Args:
            model (str): Groq whisper model to use
        """
        self.client = client
        self.model = model
    
    def transcribe_audio(self, 
                          audio_file: str, 
                          language: Optional[str] = None,
                          prompt: Optional[str] = None) -> str:
        """
        Transcribe an audio file using Groq API.
        
        Args:
            audio_file (str): Path to the audio file
            language (Optional[str]): Language of the audio
            prompt (Optional[str]): Context prompt for transcription
        
        Returns:
            str: Transcribed text
        """
        try:
            with open(audio_file, "rb") as file:
                transcription = self.client.audio.transcriptions.create(
                    file=(audio_file, file.read()),
                    model=self.model,
                    language=language,
                    prompt=prompt,
                    response_format="text"
                )
                return transcription
        except Exception as e:
            raise RuntimeError(f"Transcription failed for {audio_file}: {e}")

class AudioTranscriptionProcessor:
    """Orchestrates the entire audio transcription process."""
    
    @staticmethod
    def process_audio(input_file: str, 
                      output_file: str, 
                      language: Optional[str] = None,
                      prompt: Optional[str] = None) -> None:
        """
        Process audio file: downsample, split, transcribe, and combine.
        
        Args:
            input_file (str): Path to the input audio file
            output_file (str): Path to save the final transcription
            language (Optional[str]): Language of the audio
            prompt (Optional[str]): Context prompt for transcription
        """
        preprocessor = AudioFilePreprocessor()
        transcriber = GroqTranscriber()
        
        # Downsample audio
        downsampled_file = f"{input_file}_downsampled.mp3"
        preprocessor.downsample_audio(input_file, downsampled_file)
        
        # Split audio if necessary
        audio_chunks = preprocessor.split_audio_file(downsampled_file)
        
        # Transcribe chunks
        full_transcription = []
        for chunk in audio_chunks:
            try:
                transcription = transcriber.transcribe_audio(
                    chunk, 
                    language=language, 
                    prompt=prompt
                )
                full_transcription.append(transcription)
            except Exception as e:
                print(f"Error transcribing chunk {chunk}: {e}")
        
        # Combine transcriptions and save
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(full_transcription))
        
        # Clean up temporary files
        os.remove(downsampled_file)
        for chunk in audio_chunks:
            if chunk != input_file:
                os.remove(chunk)

def main():
    """CLI entry point for audio transcription."""
    parser = argparse.ArgumentParser(description="Transcribe audio files using Groq API")
    parser.add_argument("input", help="Input audio file path")
    parser.add_argument("output", help="Output text file path")
    parser.add_argument("--language", help="Language of the audio (ISO 639-1 code)", default=None)
    parser.add_argument("--prompt", help="Context prompt for transcription", default=None)
    
    args = parser.parse_args()
    
    try:
        AudioTranscriptionProcessor.process_audio(
            args.input, 
            args.output, 
            language=args.language,
            prompt=args.prompt
        )
        print(f"Transcription complete. Output saved to {args.output}")
    except Exception as e:
        print(f"Transcription failed: {e}")

if __name__ == "__main__":
    main()