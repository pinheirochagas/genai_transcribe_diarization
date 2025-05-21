# Define functions
#%% Import libraries
from moviepy.editor import VideoFileClip
from IPython.display import Audio
import os
import librosa
import time
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import pyannote.audio
import whisper
import soundfile as sf
import numpy as np
from tqdm import tqdm
from pyannote.audio import Pipeline
import pandas as pd
from moviepy.audio.io.AudioFileClip import AudioFileClip

#%% Define functions
def extract_audio_from_video(input_video_path, output_audio_path):
    """
    Extracts audio from video files (MOV/MP4) or converts audio files (M4A) to WAV.

    Parameters:
    input_video_path (str): The path to the input video or audio file.
    output_audio_path (str): The path to save the extracted audio file.

    Returns:
    None
    """
    # Check if input file exists
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input file not found: {input_video_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
    
    # Use ffmpeg directly to extract audio
    import subprocess
    
    cmd = [
        'ffmpeg',
        '-i', input_video_path,
        '-vn',  # No video
        '-acodec', 'pcm_s16le',  # PCM 16-bit little-endian audio codec
        '-ar', '44100',  # 44.1kHz sample rate
        '-ac', '2',  # 2 channels (stereo)
        '-y',  # Overwrite output file if it exists
        output_audio_path
    ]
    
    try:
        print(f"Extracting audio from {input_video_path} to {output_audio_path}...")
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Successfully extracted audio to: {output_audio_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e}")
        print(f"FFMPEG stderr: {e.stderr.decode()}")
        raise

def transcribe_audio(audio_path, return_timestamps=True, model_id="openai/whisper-large-v3", device="cuda:0"):
    """
    Transcribes audio to text using a speech recognition pipeline.

    Parameters:
    audio_path (str): Path to the audio file to transcribe.
    return_timestamps (bool, optional): Whether to return word-level timestamps. Defaults to False.

    Returns:
    dict or str: If return_timestamps is True, returns a dictionary containing the transcription and timestamps.
                If False, returns just the transcribed text as a string.
    """


    # Set up device and data type
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


    # Disable dynamic compilation optimizations
    torch._dynamo.config.suppress_errors = True

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation="eager"  # Use eager execution instead of optimized attention
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    # Create pipeline
    if model_id == "openai/whisper-large-v3":
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
        device=device,
        )
    else:
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            return_timestamps='word',
            device=device,
        )
    result = pipe(
        audio_path,
        return_timestamps=return_timestamps
    )
    return result

def transcribe_audio_with_diarization(audio_path, model_id="openai/whisper-large-v3"):
    """
    Transcribes audio with speaker diarization using pyannote.audio and Whisper.

    Parameters:
    audio_path (str): Path to the audio file to transcribe.
    model_id (str): HuggingFace model ID for the Whisper model

    Returns:
    dict: Dictionary containing the transcribed text with speaker labels and timestamps
    """
    # Initialize diarization pipeline
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    diarization_pipeline.to(torch.device("cuda"))

    print("Applying speaker diarization...")
    diarization = diarization_pipeline(audio_path)
    print("Speaker diarization complete!")

    # Initialize Whisper model from HuggingFace
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation="eager"
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    transcription_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    # Load the audio file
    audio, samplerate = sf.read(audio_path)

    # Initialize result dictionary
    result = {
        "text": "",
        "segments": []
    }

    # Process each speaker segment
    track_list = list(diarization.itertracks(yield_label=True))
    for turn, _, speaker in tqdm(track_list, desc="Transcribing Audio", 
                                total=len(track_list)):
        start, end = turn.start, turn.end

        # Extract and process speaker audio
        speaker_audio = audio[int(start * samplerate):int(end * samplerate)].copy()
        speaker_audio = speaker_audio.astype('float32')

        # Resample if necessary
        if samplerate != 16000:
            speaker_audio = librosa.resample(speaker_audio, orig_sr=samplerate, target_sr=16000)

        # Handle stereo audio
        if speaker_audio.ndim == 2:
            speaker_audio = speaker_audio[:, 0]

        # Save temporary audio segment
        temp_file = "temp_segment.wav"
        sf.write(temp_file, speaker_audio, 16000)

        # Transcribe segment using the pipeline
        segment_result = transcription_pipeline(temp_file)
        speaker_text = segment_result["text"].strip()

        # Remove temporary file
        os.remove(temp_file)

        # Add to full text and segments
        result["text"] += f"Speaker {speaker}: {speaker_text}\n"
        result["segments"].append({
            "speaker": speaker,
            "text": speaker_text,
            "start": start,
            "end": end
        })

    print("Transcription complete!")
    return result



#%% Transcribe audio with speaker diarization
# Import all required libraries at the top
import torch
import torch.nn.functional as F
from pyannote.audio import Pipeline
import soundfile as sf
import librosa
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from tqdm import tqdm
import concurrent.futures
import os

def transcribe_audio_with_speaker_diarization(
    audio_path, 
    model_id="openai/whisper-large-v3", 
    num_speakers=2
):
    """
    Optimized audio transcription with speaker diarization.
    
    Parameters:
    audio_path (str): Path to the audio file to transcribe
    model_id (str): HuggingFace model ID for the Whisper model
    num_speakers (int): Expected number of speakers
    
    Returns:
    dict: Dictionary containing the transcribed text with speaker labels and timestamps
    """
    # Initialize diarization pipeline
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    diarization_pipeline.to(torch.device("cuda"))
    
    print("Applying speaker diarization...")
    # Apply diarization with number of speakers
    diarization = diarization_pipeline(
        audio_path,
        num_speakers=num_speakers
    )
    print("Speaker diarization complete!")
    
    # Initialize Whisper model with full precision
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float32  # Always use full precision
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation="eager"
    )
    model.to(device)
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    transcription_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        chunk_length_s=30,
        stride_length_s=5
    )
    
    # Load and preprocess audio
    audio, samplerate = sf.read(audio_path)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)  # Convert stereo to mono
    
    # Resample if necessary
    if samplerate != 16000:
        audio = librosa.resample(audio, orig_sr=samplerate, target_sr=16000)
    
    # Initialize result dictionary
    result = {
        "text": "",
        "segments": []
    }
    
    def process_segment(segment_data):
        turn, speaker = segment_data
        start, end = turn.start, turn.end
        
        # Extract speaker audio
        speaker_audio = audio[int(start * 16000):int(end * 16000)].copy()
        
        # Process audio segment
        if len(speaker_audio) > 0:
            speaker_audio = torch.from_numpy(speaker_audio).float()
            speaker_audio = F.pad(speaker_audio, (0, 16000 - len(speaker_audio) % 16000))
            speaker_audio = speaker_audio.numpy()
            
            # Save temporary audio segment
            temp_file = f"temp_segment_{start}_{end}.wav"
            sf.write(temp_file, speaker_audio, 16000)
            
            # Transcribe segment
            try:
                segment_result = transcription_pipeline(temp_file)
                speaker_text = segment_result["text"].strip()
            except Exception as e:
                speaker_text = ""
                print(f"Error processing segment {start}-{end}: {str(e)}")
            
            os.remove(temp_file)
            
            return {
                "speaker": speaker,
                "text": speaker_text,
                "start": start,
                "end": end
            }
    
    # Process segments in parallel
    track_list = [(turn, speaker) for turn, _, speaker in diarization.itertracks(yield_label=True)]
    
    # Use fewer workers for larger segments to manage memory
    num_workers = min(4, len(track_list))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_segment = {executor.submit(process_segment, segment_data): segment_data 
                           for segment_data in track_list}
        
        segments = []
        for future in tqdm(concurrent.futures.as_completed(future_to_segment), 
                         total=len(track_list), 
                         desc="Transcribing segments"):
            try:
                segment = future.result()
                if segment and segment["text"]:
                    segments.append(segment)
            except Exception as e:
                print(f"Segment processing failed: {str(e)}")
    
    # Sort segments by start time
    segments.sort(key=lambda x: x["start"])
    
    # Combine results
    result["segments"] = segments
    result["text"] = "\n".join([f"Speaker {seg['speaker']}: {seg['text']}" 
                               for seg in segments])
    
    print("Transcription complete!")
    return result


# %%

def save_transcript_to_csv(result, output_file):
    """
    Saves the transcription results to a CSV file.

    Parameters:
    result (dict): The transcription result containing segments with speaker, text, start and end times
    output_file (str): Path to save the CSV file
    """
    # Create a DataFrame from the segments
    df = pd.DataFrame(result["segments"])
    
    # Reorder columns for better readability
    columns = ["speaker", "text", "start", "end"]
    df = df[columns]
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Transcript saved to {output_file}")

