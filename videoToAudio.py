import subprocess

# Path to the MP4 file
mp4_file = "internship.mkv"

# Path for the output MP3 file
mp3_file = "output.mp3"

# Run FFmpeg command
subprocess.run(["ffmpeg", "-i", mp4_file, mp3_file])

print("Conversion complete!")
