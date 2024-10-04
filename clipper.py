import os
import re
import argparse
import ffmpeg
from datetime import timedelta
from fuzzywuzzy import fuzz
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sanitize_filename(filename):
    sanitized = re.sub(r"[^\x00-\x7F]+", "-", filename)
    sanitized = sanitized.replace(" ", "_")
    return sanitized

def timestamp_to_seconds(timestamp):
    h, m, s_ms = timestamp.split(':')
    s, ms = s_ms.split(',')
    return int(timedelta(hours=int(h), minutes=int(m), seconds=int(s), milliseconds=int(ms)).total_seconds())

def convert_timestamp_format(timestamp):
    return timestamp.replace(',', '.')

def find_keyword_timestamp(srt_file, keyword, threshold=90):
    with open(srt_file, "r") as file:
        content = file.read()

    matches = re.finditer(
        r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*)", content
    )
    for match in matches:
        timestamp_start = match.group(1)
        subtitle_text = match.group(3)

        score = fuzz.partial_ratio(keyword.lower(), subtitle_text.lower())
        if score >= threshold:
            return timestamp_start, match

    return None, None

def clip_video(video_file, start_time, duration, output_file):
    start_time = str(timedelta(seconds=start_time))
    ffmpeg.input(video_file, ss=start_time, t=duration).output(output_file).run()

def extract_frame(video_file, timestamp, output_file):
    timestamp = convert_timestamp_format(timestamp)
    ffmpeg.input(video_file, ss=timestamp).output(output_file, vframes=1).run()

def extract_audio(video_file, start_time, duration, output_file):
    start_time = str(timedelta(seconds=start_time))
    ffmpeg.input(video_file, ss=start_time, t=duration).output(
        output_file, acodec="mp3"
    ).run()

def extract_subtitles(srt_file, start_time, duration, output_file):
    with open(srt_file, "r") as file:
        content = file.read()

    start_time = timedelta(seconds=start_time)
    end_time = start_time + timedelta(seconds=duration)

    matches = re.finditer(
        r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*)", content
    )
    with open(output_file, "w") as file:
        for match in matches:
            timestamp_start = match.group(1)
            timestamp_end = match.group(2)
            subtitle_text = match.group(3)

            timestamp_start_seconds = timestamp_to_seconds(timestamp_start)
            timestamp_start_timedelta = timedelta(seconds=timestamp_start_seconds)

            if start_time <= timestamp_start_timedelta <= end_time:
                file.write(f"{timestamp_start} --> {timestamp_end}\n{subtitle_text}\n\n")

def process_videos(input_folder, output_folder, keywords, pre_duration, post_duration):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".mp4"):
                video_file = os.path.join(root, file)
                srt_file = os.path.splitext(video_file)[0] + ".srt"
                if not os.path.exists(srt_file):
                    logging.warning(f"Missing .srt file for {video_file}. Skipping.")
                    continue

                for keyword in keywords:
                    try:
                        timestamp, match = find_keyword_timestamp(srt_file, keyword)
                        if timestamp is None:
                            logging.info(f"Keyword '{keyword}' not found in {srt_file}.")
                            continue

                        timestamp_seconds = timestamp_to_seconds(timestamp)
                        start_time = max(0, timestamp_seconds - pre_duration)
                        duration = pre_duration + post_duration

                        sanitized_filename = sanitize_filename(file)
                        output_base = os.path.join(output_folder, f"{sanitized_filename}_{keyword}_{timestamp}")

                        # Clip video
                        clip_video(video_file, start_time, duration, f"{output_base}.mp4")

                        # Extract frame
                        extract_frame(video_file, timestamp, f"{output_base}.jpg")

                        # Extract audio
                        extract_audio(video_file, start_time, duration, f"{output_base}.mp3")

                        # Extract subtitles
                        extract_subtitles(srt_file, start_time, duration, f"{output_base}.srt")

                        # Write metadata
                        metadata = {
                            "original_file": file,
                            "keyword": keyword,
                            "timestamp": timestamp,
                            "start_time": start_time,
                            "duration": duration,
                        }
                        with open(f"{output_base}.json", "w") as json_file:
                            json.dump(metadata, json_file)

                        # Write center point and keyword to .txt file
                        with open(f"{output_base}.txt", "w") as txt_file:
                            txt_file.write(f"Center point: {timestamp}\n")
                            txt_file.write(f"Keyword: {keyword}\n")

                    except Exception as e:
                        logging.error(f"Error processing {video_file} with keyword '{keyword}': {e}")

def main():
    """
    Parses command-line arguments and initiates the video processing.
    """
    parser = argparse.ArgumentParser(
        description="Clip videos based on keywords in subtitles."
    )
    parser.add_argument(
        "-i",
        "--input-folder",
        default="./mediaSource",
        help="The folder containing the video files and accompanying .srt files.",
    )
    parser.add_argument(
        "-o",
        "--output-folder",
        default="./mediaClips",
        help="The folder to output the clipped files.",
    )
    parser.add_argument(
        "-k",
        "--keywords",
        nargs="+",
        default=["test", "demo"],
        help="The keywords to search for in the .srt files.",
    )
    parser.add_argument(
        "-p",
        "--pre-duration",
        type=int,
        default=10,
        help="The duration in seconds before the center point of the clip.",
    )
    parser.add_argument(
        "-t",
        "--post-duration",
        type=int,
        default=10,
        help="The duration in seconds after the center point of the clip.",
    )

    args = parser.parse_args()

    process_videos(
        args.input_folder,
        args.output_folder,
        args.keywords,
        args.pre_duration,
        args.post_duration,
    )

if __name__ == "__main__":
    main()

