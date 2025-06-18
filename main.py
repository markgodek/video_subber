import os, ffmpeg, whisper, torch

input_file = r"C:\Users\markg\Downloads\[HnY] Beyblade X 79 - The King and the Queen (1080p).mkv"
output_file = 'output_audio.wav'

print(input_file)

# Extract audio using ffmpeg-python
if os.path.isfile(output_file):
    print(f"Output file already exists: {output_file}, skipping extraction.")
else:
    ffmpeg.input(input_file).output(output_file, ac=1, ar=16000).run()
    print(f"Audio extracted to {output_file}")

# Load Whisper model and transcribe (with translation from Japanese to English)
model = whisper.load_model("small")  # You can change this to "medium" or "large" for better accuracy
result = model.transcribe(output_file, task="translate", language="ja")

print("\nTranslated English Text:\n")
print(result["text"])
