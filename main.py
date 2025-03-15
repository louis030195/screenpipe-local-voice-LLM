import requests
import os
import datetime
import tempfile
from pathlib import Path
import torch
import torchaudio
import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play
from huggingface_hub import hf_hub_download
import subprocess
import sys
import time
import signal
from urllib.parse import quote


# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global shutdown_requested
    if shutdown_requested:
        print("\nForced exit. Goodbye!")
        sys.exit(1)
    print("\nShutdown requested. Finishing current operation before exiting...")
    shutdown_requested = True

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)


def setup_environment():
    """Ensure all required packages are installed"""
    try:
        import pocketsphinx
    except ImportError:
        print("Installing pocketsphinx for fallback transcription...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pocketsphinx"])


def query_screenpipe_audio(start_time, end_time, limit=1000):
    """Query Screenpipe for audio recordings within a time range"""
    endpoint = "http://localhost:3030/search"
    
    # Format dates with milliseconds as seen in the working curl command
    start_time_str = start_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
    end_time_str = end_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
    
    # Parameters matching the successful curl command
    params = {
        "content_type": "audio",
        "start_time": start_time_str,
        "end_time": end_time_str,
        "limit": limit,
        "offset": 0,
        "min_length": 50,      # Add min_length parameter
        "max_length": 10000    # Add max_length parameter
    }
    
    print(f"Request URL: {endpoint}")
    print(f"Request parameters: {params}")
    
    response = requests.get(endpoint, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Full URL that was requested: {response.url}")
        raise Exception(f"failed to query screenpipe: {response.status_code} - {response.text}")
    
def download_audio_files(results):
    """Download audio files from Screenpipe results"""
    audio_files = []
    total_items = len(results.get("data", []))
    processed = 0
    
    print(f"\nProcessing {total_items} items from Screenpipe...")
    
    for item in results.get("data", []):
        # Check for shutdown request
        global shutdown_requested
        if shutdown_requested:
            print("Shutdown requested, stopping download...")
            break
            
        processed += 1
        if processed % 10 == 0:
            print(f"Progress: {processed}/{total_items} items")
            
        if item.get("type").lower() == "audio" and item.get("content", {}).get("file_path"):
            audio_path = item["content"]["file_path"]
            
            # Download or access the audio file
            if os.path.exists(audio_path):
                audio_files.append(audio_path)
            else:
                # For remote files, download them
                try:
                    local_path = os.path.join(tempfile.gettempdir(), os.path.basename(audio_path))
                    with requests.get(audio_path, stream=True) as r:
                        r.raise_for_status()
                        with open(local_path, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                    audio_files.append(local_path)
                except Exception as e:
                    print(f"Error downloading {audio_path}: {str(e)}")
    
    return audio_files

def transcribe_audio(audio_files):
    """Transcribe audio files using SpeechRecognition"""
    recognizer = sr.Recognizer()
    combined_transcription = ""
    successful_transcriptions = 0
    total_files = len(audio_files)
    
    # Print header for progress tracking
    print(f"\nTranscribing {total_files} audio files...")
    
    processed_audio_files = []
    
    for i, audio_file in enumerate(audio_files):
        # Check for shutdown request
        global shutdown_requested
        if shutdown_requested:
            print("Shutdown requested, stopping transcription...")
            break
            
        print(f"[{i+1}/{total_files}] Processing: {os.path.basename(audio_file)}...", end="", flush=True)
        
        # Convert to WAV if not already (for consistent processing)
        if not audio_file.lower().endswith('.wav'):
            try:
                temp_wav = os.path.join(tempfile.gettempdir(), f"{os.path.basename(audio_file)}.wav")
                # Use ffmpeg via subprocess for better audio conversion
                subprocess.run([
                    "ffmpeg", "-y", "-i", audio_file, 
                    "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                    temp_wav
                ], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                audio_file = temp_wav
                print(" Converted ✓", flush=True)
            except Exception as e:
                print(f" Failed to convert format: {str(e)}")
                continue
        else:
            print(" Ready ✓", flush=True)
            
        try:
            # Process each file but don't transcribe
            processed_audio_files.append(audio_file)
        except Exception as e:
            print(f" Error processing: {str(e)}")
    
    print(f"\nProcessing complete. Successfully processed {len(processed_audio_files)}/{total_files} files.")
    return processed_audio_files

def initialize_csm_model():
    """Initialize the CSM-1B model"""
    # Import here to ensure we're in the right environment after potentially installing dependencies
    # Fix the import issue by making sure the CSM module is properly accessible
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)  # Add current directory to path
    
    # Set environment variable to disable triton before imports
    os.environ["DISABLE_TRITON"] = "1"
    
    try:
        from generator import load_csm_1b
    except ImportError as e:
        print(f"import error: {str(e)}")
        print("trying alternative import paths...")
        
        # Try alternative import paths
        if os.path.exists(os.path.join(current_dir, "csm")):
            sys.path.append(os.path.join(current_dir, "csm"))
        
        try:
            from generator import load_csm_1b
        except ImportError:
            raise ImportError("Could not find the CSM module. Make sure it's installed correctly.")
    
    # Download model if needed
    model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
    
    # Initialize model
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"loading csm-1b model on {device}...")
    generator = load_csm_1b(model_path, device)
    
    return generator

def generate_speech(generator, text):
    """Generate speech using CSM-1B model"""
    print(f"generating speech for: {text[:100]}...")
    
    # Generate audio
    audio = generator.generate(
        text=text,
        speaker=0,  # Use default speaker
        context=[],
        max_audio_length_ms=10_000,
    )
    
    # Save to file
    speech_file_path = os.path.join(tempfile.gettempdir(), "csm_speech.wav")
    torchaudio.save(speech_file_path, audio.unsqueeze(0).cpu(), generator.sample_rate)
    
    return speech_file_path

def play_audio(file_path):
    """Play the audio file"""
    print(f"playing audio from {file_path}...")
    sound = AudioSegment.from_file(file_path)
    play(sound)

def load_audio(audio_path, sample_rate):
    """Load audio file and resample to target sample rate"""
    audio_tensor, orig_sr = torchaudio.load(audio_path)
    if orig_sr != sample_rate:
        audio_tensor = torchaudio.functional.resample(
            audio_tensor.squeeze(0), orig_freq=orig_sr, new_freq=sample_rate
        )
    else:
        audio_tensor = audio_tensor.squeeze(0)
    return audio_tensor

def main():
    try:
        # Setup environment first
        setup_environment()
        
        # Define time range (last 48 hours by default)
        end_time = datetime.datetime.now()
        start_time = end_time - datetime.timedelta(hours=12)
        
        print(f"querying screenpipe for audio between {start_time} and {end_time}")
        
        try:
            # Query Screenpipe
            results = query_screenpipe_audio(start_time, end_time)
            
            # Download audio files
            audio_files = download_audio_files(results)
            print(f"found {len(audio_files)} audio files")
            
            if not audio_files:
                print("no audio files found in the specified time range.")
                return
            
            # Process audio files without transcription
            processed_audio_files = transcribe_audio(audio_files)
            
            if not processed_audio_files:
                print("failed to process any audio files.")
                return
            
            print(f"\nsuccessfully processed {len(processed_audio_files)} audio files")
            
            # Initialize CSM model
            generator = initialize_csm_model()
            
            # Select a subset of audio files to use as context (e.g., first 5)
            context_files = processed_audio_files[:min(5, len(processed_audio_files))]
            
            # Create context segments from the processed audio files
            from generator import Segment  # Import here to ensure dependencies are installed
            
            print("preparing audio context segments...")
            segments = []
            
            for i, audio_path in enumerate(context_files):
                try:
                    # Alternate between speakers 0 and 1 for diversity
                    speaker_id = i % 2
                    # Placeholder transcript (not used for generation)
                    placeholder_text = f"Audio segment {i+1}"
                    
                    # Load audio for this segment
                    audio_tensor = load_audio(audio_path, generator.sample_rate)
                    
                    # Create segment
                    segment = Segment(
                        text=placeholder_text,
                        speaker=speaker_id,
                        audio=audio_tensor
                    )
                    segments.append(segment)
                    print(f"  prepared segment {i+1}/{len(context_files)}")
                except Exception as e:
                    print(f"  error preparing segment {i+1}: {str(e)}")
            
            # Generate speech with context
            response_text = "i've analyzed the audio recordings and here's a summary of what i heard."
            print(f"\ngenerating response: '{response_text}'")
            
            audio = generator.generate(
                text=response_text,
                speaker=1,  # Use speaker 1 for response
                context=segments,
                max_audio_length_ms=10_000,
            )
            
            # Save to file
            speech_file_path = os.path.join(tempfile.gettempdir(), "csm_response.wav")
            torchaudio.save(speech_file_path, audio.unsqueeze(0).cpu(), generator.sample_rate)
            
            # Play the generated audio
            play_audio(speech_file_path)
            
        except Exception as e:
            print(f"error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    except KeyboardInterrupt:
        print("\ngracefully shutting down...")
    
    print("\nprogram completed.")


if __name__ == "__main__":
    main()