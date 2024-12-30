import os
import whisper

# Directories
input_directory = r"E:\ADAM_NU"  # Change to your actual path
output_directory = r"E:\ADAM_NU\output"  # Change to your actual path
os.makedirs(output_directory, exist_ok=True)

# Load the Whisper model
model = whisper.load_model("medium")

# Get all .mp3 files
mp3_files = [f for f in os.listdir(input_directory) if f.endswith('.mp3')]

# Function to format time in SRT style
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{int(seconds):02},{milliseconds:03}"

# Transcribe each file
for filename in mp3_files:
    mp3_path = os.path.join(input_directory, filename)
    print(f"Processing: {mp3_path}")

    # Perform transcription
    result = model.transcribe(mp3_path, language="en")

    # Save transcription as SRT
    file_id = os.path.splitext(filename)[0]
    srt_path = os.path.join(output_directory, f"{file_id}.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result['segments']):
            start = format_time(segment['start'])
            end = format_time(segment['end'])
            text = segment['text'].strip()
            
            f.write(f"{i + 1}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")
    
    print(f"Transcription saved to: {srt_path}")

print("All files processed!")
