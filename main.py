import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # avoid OpenMP DLL clash on Windows

import whisper
import torch
import ffmpeg
import argparse
import warnings
import time
import glob
import sys

warnings.filterwarnings("ignore", category=FutureWarning)

# toggle settings
translate = True
transcribe = False

def write_srt(segments, path):
    def format_timestamp(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{hours:02}:{minutes:02}:{int(seconds):02},{milliseconds:03}"

    with open(path, "w", encoding="utf-8") as srt_file:
        for i, segment in enumerate(segments, start=1):
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment["text"].strip()
            srt_file.write(f"{i}\n{start} --> {end}\n{text}\n\n")

def write_transcript(segments, path):
    with open(path, "w", encoding="utf-8") as txt_file:
        for segment in segments:
            txt_file.write(segment["text"].strip() + "\n")

def main():
    start_time = time.perf_counter()  # Start timer

    parser = argparse.ArgumentParser(description="Video to translated subtitles CLI using Whisper")
    parser.add_argument("input_file", nargs="?", help="Path to input video file")  # optional arg
    args = parser.parse_args()

    if args.input_file:
        # Use provided file
        input_file = args.input_file
    else:
        # Fallback: search for the most recent beyblade file
        download_dir = r"C:\Users\markg\Downloads"
        video_extensions = ["*.mkv", "*.mp4", "*.avi", "*.mov", "*.wmv", "*.flv", "*.webm"]

        matching_files = []
        for ext in video_extensions:
            pattern = os.path.join(download_dir, f"*beyblade*{ext}")
            matching_files.extend(glob.glob(pattern))

        if not matching_files:
            raise FileNotFoundError(f"No video files with 'beyblade' found in {download_dir}")

        input_file = max(matching_files, key=os.path.getmtime)

    output_audio = 'output_audio.wav'
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    base_dir = os.path.dirname(input_file)

    srt_output = os.path.join(base_dir, base_name + ".srt")
    transcript_output = os.path.join(base_dir, base_name + " - transcript.txt")

    print(f"Input file: {input_file}")

    if os.path.isfile(output_audio):
        print(f"Output audio file already exists: {output_audio}, skipping extraction.")
    else:
        print("Extracting audio from video...")
        ffmpeg.input(input_file).output(output_audio, ac=1, ar=16000).run()
        print(f"Audio extracted to {output_audio}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if translate:
        model = whisper.load_model("large").to(device)

        print("Translating audio to English...")
        en_result = model.transcribe(output_audio, task="translate", language="ja")

        write_srt(en_result["segments"], srt_output)
        print(f"✅ SRT file saved as {srt_output}")

    if transcribe:
        model = whisper.load_model("turbo").to(device) # turbo model does not support translation

        print("Transcribing audio in Japanese...")
        ja_result = model.transcribe(output_audio, task="transcribe", language="ja")

        write_transcript(ja_result["segments"], transcript_output)
        print(f"✅ Japanese transcript file saved as {transcript_output}")

    if os.path.exists(output_audio):
        try:
            os.remove(output_audio)
            print(f"Deleted temporary file: {output_audio}")
        except Exception as e:
            print(f"Could not delete {output_audio}: {e}")

    elapsed = time.perf_counter() - start_time  # Compute elapsed time
    print(f"\n⏱ Total execution time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
