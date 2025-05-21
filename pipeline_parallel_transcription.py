#%% Imports
import os
import sys
import time
from multiprocessing import Pool
import pandas as pd

from transcription_utils import (
    transcribe_audio_with_speaker_diarization,
    save_transcript_to_csv
)

AUDIO_DIR = "/audios"
VIDEO_LIST_PATH = "/audio_files_process.csv"
TRANSCRIPT_DIR = "/transcripts"

#%% Read audio files list and start parallel processing
print("\nStarting parallel transcription processing...")
start_time = time.time()

def process_single_audio(row):
    """Process a single audio file with transcription and saving to CSV"""
    audio_file = row['file_name_audio']
    num_speakers = row['number_of_voices']
    
    # Check if transcript already exists
    output_path = os.path.join(TRANSCRIPT_DIR, f"{os.path.splitext(audio_file)[0]}_transcript.csv")
    if os.path.exists(output_path):
        print(f"\nSkipping {audio_file} - transcript already exists")
        return {"audio_file": audio_file, "status": "skipped"}
    
    print(f"\nProcessing {audio_file}...")
    try:
        # Perform transcription
        result_with_speakers = transcribe_audio_with_speaker_diarization(
            os.path.join(AUDIO_DIR, audio_file),
            num_speakers=num_speakers
        )
        
        # Save to CSV with audio filename as part of the output path
        save_transcript_to_csv(result_with_speakers, output_path)
        
        return {"audio_file": audio_file, "status": "success"}
    except Exception as e:
        print(f"Error processing {audio_file}: {str(e)}")
        return {"audio_file": audio_file, "status": "error", "error": str(e)}

#%% Read audio files list
video_list_df = pd.read_csv(VIDEO_LIST_PATH)

# Create a process pool with 8 processes
with Pool(processes=3) as pool:
    # Map the processing function to all rows in the DataFrame
    results = pool.map(process_single_audio, [row for _, row in video_list_df.iterrows()])

end_time = time.time()
print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")

# Create a summary of results
results_df = pd.DataFrame(results)
print("\nProcessing Summary:")
print(f"Successfully processed: {sum(results_df['status'] == 'success')} files")
print(f"Skipped existing: {sum(results_df['status'] == 'skipped')} files")
print(f"Failed: {sum(results_df['status'] == 'error')} files")

# Export failed files and their errors to CSV
failed_files = results_df[results_df['status'] == 'error']
if len(failed_files) > 0:
    error_log_path = os.path.join(TRANSCRIPT_DIR, 'failed_transcriptions.csv')
    failed_files.to_csv(error_log_path, index=False)
    print(f"\nFailed files and errors have been saved to: {error_log_path}")
# %%
