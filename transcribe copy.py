import os
import io
import tempfile
import subprocess
import argparse
from typing import List, Optional, Union, Dict, Any

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize AWS clients
s3_client = boto3.client('s3')
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class S3FileHandler:
    """Handles file operations with Amazon S3."""
    
    @staticmethod
    def download_file(bucket: str, key: str) -> str:
        """
        Download a file from S3 to a temporary local file.
        
        Args:
            bucket (str): S3 bucket name
            key (str): S3 object key
        
        Returns:
            str: Path to the downloaded temporary file
        """
        try:
            # Create a temporary file
            temp_file = tempfile.mktemp()
            
            # Download file from S3
            s3_client.download_file(bucket, key, temp_file)
            
            return temp_file
        except ClientError as e:
            raise RuntimeError(f"S3 download failed: {e}")
    
    @staticmethod
    def upload_file(local_path: str, bucket: str, key: str) -> None:
        """
        Upload a local file to S3.
        
        Args:
            local_path (str): Path to the local file
            bucket (str): S3 bucket name
            key (str): S3 object key
        """
        try:
            s3_client.upload_file(local_path, bucket, key)
        except ClientError as e:
            raise RuntimeError(f"S3 upload failed: {e}")
    
    @staticmethod
    def upload_text(text: str, bucket: str, key: str) -> None:
        """
        Upload text content directly to S3.
        
        Args:
            text (str): Text content to upload
            bucket (str): S3 bucket name
            key (str): S3 object key
        """
        try:
            s3_client.put_object(Bucket=bucket, Key=key, Body=text.encode('utf-8'))
        except ClientError as e:
            raise RuntimeError(f"S3 text upload failed: {e}")

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
        self.client = groq_client
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
    
    @classmethod
    def process_audio(cls, 
                      input_source: Union[str, Dict[str, Any]], 
                      output_destination: Union[str, Dict[str, Any]], 
                      language: Optional[str] = None,
                      prompt: Optional[str] = None,
                      source_type: str = 'local') -> None:
        """
        Process audio file: download, downsample, split, transcribe, and upload.
        
        Args:
            input_source (Union[str, Dict]): Input file path or S3 details
            output_destination (Union[str, Dict]): Output file path or S3 details
            language (Optional[str]): Language of the audio
            prompt (Optional[str]): Context prompt for transcription
            source_type (str): Source type - 'local' or 's3'
        """
        preprocessor = AudioFilePreprocessor()
        transcriber = GroqTranscriber()
        
        # Handle input based on source type
        if source_type == 's3':
            # Download from S3
            input_file = S3FileHandler.download_file(
                input_source['bucket'], 
                input_source['key']
            )
        else:
            input_file = input_source
        
        try:
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
            
            # Combine transcriptions
            combined_transcription = '\n\n'.join(full_transcription)
            
            # Handle output based on destination type
            if source_type == 's3':
                # Upload to S3
                S3FileHandler.upload_text(
                    combined_transcription,
                    output_destination['bucket'], 
                    output_destination['key']
                )
            else:
                # Save to local file
                with open(output_destination, 'w', encoding='utf-8') as f:
                    f.write(combined_transcription)
        
        finally:
            # Clean up temporary files
            if os.path.exists(downsampled_file):
                os.remove(downsampled_file)
            for chunk in audio_chunks:
                if chunk != input_file and os.path.exists(chunk):
                    os.remove(chunk)
            if source_type == 's3' and os.path.exists(input_file):
                os.remove(input_file)

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler for audio transcription.
    
    Expected event structure:
    {
        "input_bucket": "source-bucket",
        "input_key": "path/to/audio/file.mp3",
        "output_bucket": "destination-bucket",
        "output_key": "path/to/transcription/output.txt",
        "language": "en",  # Optional
        "prompt": "Technical conference"  # Optional
    }
    """
    try:
        # Extract parameters from event
        input_bucket = event.get('input_bucket')
        input_key = event.get('input_key')
        output_bucket = event.get('output_bucket')
        output_key = event.get('output_key')
        language = event.get('language')
        prompt = event.get('prompt')
        
        # Validate required parameters
        if not all([input_bucket, input_key, output_bucket, output_key]):
            raise ValueError("Missing required S3 bucket or key parameters")
        
        # Process audio
        AudioTranscriptionProcessor.process_audio(
            input_source={'bucket': input_bucket, 'key': input_key},
            output_destination={'bucket': output_bucket, 'key': output_key},
            language=language,
            prompt=prompt,
            source_type='s3'
        )
        
        return {
            'statusCode': 200,
            'body': f'Transcription complete. Output saved to {output_bucket}/{output_key}'
        }
    
    except Exception as e:
        print(f"Transcription error: {e}")
        return {
            'statusCode': 500,
            'body': f'Transcription failed: {str(e)}'
        }

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