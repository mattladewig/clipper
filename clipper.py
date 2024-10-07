import os
import re
import logging
import argparse
import concurrent.futures
import json
import threading
from datetime import timedelta
from fuzzywuzzy import fuzz
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import ffmpeg
import nltk
from nltk.stem import WordNetLemmatizer as wnl
from nltk.corpus import wordnet
from word_alt_map import word_alt_map


class ProcessMetaThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.daemon = True  # Set the daemon attribute directly

    def run(self):
        lemmatizer = wnl()
        lemmatizer.lemmatize("dog")


def generate_bidirectional_mapping(base_dict):
    bidirectional_dict = {}
    for key, values in base_dict.items():
        if key not in bidirectional_dict:
            bidirectional_dict[key] = set(values)
        for value in values:
            if value not in bidirectional_dict:
                bidirectional_dict[value] = set()
            bidirectional_dict[value].add(key)
            bidirectional_dict[value].update(values)
    # Convert sets back to lists
    for key in bidirectional_dict:
        bidirectional_dict[key] = list(bidirectional_dict[key])
    return bidirectional_dict


# Generate the bidirectional mapping
custom_alternatives = generate_bidirectional_mapping(word_alt_map)

# Ensure nltk resources are downloaded
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

nltk.data.path.append(nltk_data_path)

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", download_dir=nltk_data_path)

try:
    nltk.data.find("corpora/omw-1.4")
except LookupError:
    nltk.download("omw-1.4", download_dir=nltk_data_path)


# Create log directory if it doesn't exist
log_directory = "./log"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# Configure logging to write to a file in the log directory
log_file = os.path.join(log_directory, "clipper.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=log_file,
    filemode="a",
)


def get_word_forms(word):
    """
    Generate different forms of a word: present participle, past tense, possessive, and common alternatives.
    Args:
        word (str): The original word.
    Returns:
        list: A list of word forms.
    """
    word_forms = [word]
    lemmatizer = wnl()

    # Add common alternatives
    if word in custom_alternatives:
        word_forms.extend(custom_alternatives[word])

    # Generate present participle and past tense forms
    present_participle = lemmatizer.lemmatize(word, "v") + "ing"
    past_tense = lemmatizer.lemmatize(word, "v") + "ed"
    # Generate possessive and plural forms for nouns
    possessive_noun = lemmatizer.lemmatize(word, "n") + "'s"
    plural_noun = lemmatizer.lemmatize(word, "n") + "s"
    word_forms.extend([present_participle, past_tense, possessive_noun, plural_noun])

    # Remove duplicates
    word_forms = list(set(word_forms))
    logging.info(
        f"Will search keyword '{word}' and generative alternatives: {word_forms}"
    )
    return word_forms


def sanitize_filename(filename):
    """
    Sanitizes a given filename by removing non-ASCII characters and replacing spaces with underscores.
    Args:
        filename (str): The original filename to be sanitized.
    Returns:
        str: The sanitized filename with non-ASCII characters replaced by hyphens and spaces replaced by underscores.
    """

    sanitized = re.sub(r"[^\x00-\x7F]+", "-", filename)
    sanitized = sanitized.replace(" ", "_")
    return sanitized


def timestamp_to_seconds(timestamp):
    """
    Convert a timestamp in the format 'HH:MM:SS,mmm' to total seconds, ignoring milliseconds.
    Args:
        timestamp (str): The timestamp string to convert, in the format 'HH:MM:SS,mmm'.
    Returns:
        int: The total number of seconds represented by the timestamp.
    """
    if isinstance(timestamp, int):
        timestamp = str(timedelta(seconds=timestamp))
    h, m, s = timestamp.split(":")
    s = s.split(",")[0]  # Ignore milliseconds
    return int(h) * 3600 + int(m) * 60 + int(s)


def find_keyword_timestamps(srt_file, keyword, threshold):
    """
    Find timestamps in an SRT file where a given keyword appears in the transcript.
    Args:
        srt_file (str): Path to the SRT file.
        keyword (str): The keyword to search for in the transcript.
        threshold (int): The minimum similarity score (0-100) for a word to be considered a match.
    Returns:
        list of tuple: A list of tuples where each tuple contains:
            - timestamp_start (str): The start timestamp of the subtitle segment (HH:MM:SS).
            - transcript_text (str): The text of the subtitle segment.
            - score (int): The similarity score of the matched word.
    """
    with open(srt_file, "r") as file:
        content = file.read()

    matches = re.finditer(
        r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*)", content
    )
    results = []
    word_forms = get_word_forms(keyword)

    for match in matches:
        timestamp_start = match.group(1).split(",")[0]  # Ignore milliseconds
        timestamp_end = match.group(2).split(",")[0]  # Ignore milliseconds
        transcript_text = match.group(3)

        words = transcript_text.split()
        for word in words:
            for form in word_forms:
                score = fuzz.ratio(form.lower(), word.lower())
                if score >= threshold:
                    results.append((timestamp_start, transcript_text, score))
                    break

    return results


def clip_video(
    video_file, timestamp_start, timestamp_end, pre_duration, post_duration, output_file
):
    """
    Clips a segment from a video file based on the provided timestamps and durations.

    Args:
        video_file (str): Path to the input video file.
        timestamp_start (str): Start timestamp in the format 'HH:MM:SS'.
        timestamp_end (str): End timestamp in the format 'HH:MM:SS'.
        pre_duration (int): Duration in seconds to include before the start timestamp.
        post_duration (int): Duration in seconds to include after the end timestamp.
        output_file (str): Path to the output video file.

    Returns:
        None
    """
    start_time = max(0, timestamp_to_seconds(timestamp_start) - pre_duration)
    end_time = timestamp_to_seconds(timestamp_end) + post_duration
    logging.info(f"Clipping video from {start_time} to {end_time}")
    ffmpeg_extract_subclip(video_file, start_time, end_time, targetname=output_file)


def extract_frame(video_file, timestamp, output_file):
    """
    Extracts a single frame from a video file at a specified timestamp and saves it as an image.

    Args:
        video_file (str): Path to the input video file.
        timestamp (str): Timestamp in the format 'HH:MM:SS' to extract the frame from.
        output_file (str): Path to save the extracted frame image.

    Returns:
        None

    Raises:
        ffmpeg.Error: If there is an error during the ffmpeg command execution.

    Example:
        extract_frame('input.mp4', '00:01:23', 'output.png')
    """
    logging.info(
        f"Extracting frame: {video_file}, timestamp: {timestamp}, output_file: {output_file}"
    )
    timestamp = timestamp.split(",")[0]  # Ignore milliseconds
    logging.info(
        f"Running ffmpeg command: ffmpeg.input({video_file}, ss={timestamp}).output({output_file}, vframes=1, format='image2', vcodec='png').run()"
    )
    ffmpeg.input(video_file, ss=timestamp).output(
        output_file, vframes=1, format="image2", vcodec="png"
    ).run()


def extract_frame(video_file, timestamp, output_file):
    """
    Extracts a frame from a video file at a specific timestamp and saves it as a PNG file.
    Args:
        video_file (str): Path to the input video file.
        timestamp (str or int): Timestamp in the format 'HH:MM:SS' or seconds from which to extract the frame.
        output_file (str): Path to the output PNG file.

    Returns:
        None

    Raises:
        ffmpeg.Error: If there is an error during the ffmpeg command execution.

    Example:
        extract_frame('input.mp4', '00:01:23', 'output.png')
    """
    logging.info(
        f"Extracting frame: {video_file}, timestamp: {timestamp}, output_file: {output_file}"
    )

    # Ensure timestamp is a string
    if isinstance(timestamp, int):
        timestamp = str(timedelta(seconds=timestamp))
    else:
        timestamp = timestamp.split(",")[0]  # Ignore milliseconds if present

    logging.info(
        f"Running ffmpeg command: ffmpeg.input({video_file}, ss={timestamp}).output({output_file}, vframes=1, format='image2', vcodec='png').run()"
    )
    ffmpeg.input(video_file, ss=timestamp).output(
        output_file, vframes=1, format="image2", vcodec="png"
    ).run()


def extract_audio(video_file, start_time, duration, output_file):
    """
    Extracts a segment of audio from a video file and saves it as an MP3 file.
    Args:
        video_file (str): Path to the input video file.
        start_time (int): Start time in seconds from which to begin extracting audio.
        duration (int): Duration in seconds of the audio segment to extract.
        output_file (str): Path to the output MP3 file.
    Returns:
        None
    Raises:
        ffmpeg.Error: If there is an error during the ffmpeg processing.
    """
    logging.info(
        f"Extracting audio: {video_file}, start_time: {start_time}, duration: {duration}, output_file: {output_file}"
    )
    start_time_str = str(timedelta(seconds=start_time))
    logging.info(f"Formatted start_time: {start_time_str}")
    logging.info(
        f"Running ffmpeg command: ffmpeg.input({video_file}, ss={start_time_str}, t={duration}).output({output_file}, acodec='mp3').run()"
    )
    ffmpeg.input(video_file, ss=start_time_str, t=duration).output(
        output_file, acodec="mp3"
    ).run()


def extract_transcript(srt_file, start_time, duration, output_file):
    """
    Extracts a portion of the transcript from an SRT file based on the specified start time and duration,
    and writes it to an output file.

    Args:
        srt_file (str): Path to the input SRT file.
        start_time (int): Start time in seconds from the beginning of the video.
        duration (int): Duration in seconds for which the transcript should be extracted.
        output_file (str): Path to the output file where the extracted transcript will be written.

    Returns:
        None
    """
    with open(srt_file, "r") as file:
        content = file.read()

    start_time_td = timedelta(seconds=start_time)
    end_time_td = start_time_td + timedelta(seconds=duration)

    matches = re.finditer(
        r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*)", content
    )
    with open(output_file, "w") as out_file:
        for match in matches:
            timestamp_start = match.group(1).split(",")[0]  # Ignore milliseconds
            timestamp_end = match.group(2).split(",")[0]  # Ignore milliseconds
            transcript_text = match.group(3)

            timestamp_start_td = timedelta(
                seconds=timestamp_to_seconds(timestamp_start)
            )
            timestamp_end_td = timedelta(seconds=timestamp_to_seconds(timestamp_end))

            if start_time_td <= timestamp_start_td <= end_time_td:
                out_file.write(
                    f"{timestamp_start} --> {timestamp_end}\n{transcript_text}\n\n"
                )


def get_video_duration(video_file):
    """
    Get the duration of a video file in seconds.
    Args:
        video_file (str): The path to the video file.
    Returns:
        int: The duration of the video in seconds.
    """
    probe = ffmpeg.probe(video_file)
    duration = float(probe["format"]["duration"])
    return int(duration)


def get_srt_duration(srt_file):
    """
    Calculate the duration of an SRT (SubRip Subtitle) file in seconds.
    This function reads an SRT file, extracts the timestamps of the first and last subtitles,
    and calculates the duration between them in seconds.
    Args:
        srt_file (str): The path to the SRT file.
    Returns:
        int: The duration of the SRT file in seconds. Returns 0 if no valid timestamps are found.
    """
    with open(srt_file, "r") as file:
        content = file.read()

    matches = re.findall(
        r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})", content
    )
    if not matches:
        return 0

    start_timestamp = matches[0][0].split(",")[0]  # Ignore milliseconds
    end_timestamp = matches[-1][1].split(",")[0]  # Ignore milliseconds

    start_seconds = timestamp_to_seconds(start_timestamp)
    end_seconds = timestamp_to_seconds(end_timestamp)

    return end_seconds - start_seconds


def process_videos_parallel(
    source_folder,
    output_folder,
    keywords,
    pre_duration,
    post_duration,
    threshold,
    include_other_files,
):
    """
    Processes video files in parallel by extracting clips based on provided keywords and durations.

    Args:
        source_folder (str): Path to the folder containing the source video files.
        output_folder (str): Path to the folder where the processed video clips will be saved.
        keywords (list): List of keywords to search for in the video subtitles.
        pre_duration (int): Duration (in seconds) to include before the keyword occurrence.
        post_duration (int): Duration (in seconds) to include after the keyword occurrence.
        threshold (float): Confidence threshold for keyword matching.
        include_other_files (bool): Flag to include other related files in the output folder.

    Raises:
        Exception: If an error occurs during video processing.

    Notes:
        - The function ensures that the WordNet resource is loaded before processing.
        - It creates the output folder if it does not exist.
        - It collects all .mp4 video files from the source folder and its subdirectories.
        - It initializes a number of threads to ensure that the WordNetLemmatizer is thread-safe.
        - It processes the videos in parallel using a ThreadPoolExecutor.
        - It logs a warning if a corresponding .srt file is not found for a video.
        - It waits for all threads to complete before finishing.
    """
    # Ensure WordNet is loaded
    wordnet.ensure_loaded()

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Collect video files
    video_files = []
    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.endswith(".mp4"):
                video_files.append(os.path.join(root, file))

    # Initialize threads to ensure WordNetLemmatizer is thread-safe
    threadsList = []
    numberOfThreads = 10  # Adjust the number of threads as needed
    for i in range(numberOfThreads):
        t = ProcessMetaThread()
        t.start()
        threadsList.append(t)

    # Process videos in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for video_file in video_files:
            srt_file = video_file.replace(".mp4", ".srt")
            if os.path.exists(srt_file):
                futures.append(
                    executor.submit(
                        process_video,
                        video_file,
                        srt_file,
                        keywords,
                        pre_duration,
                        post_duration,
                        threshold,
                        output_folder,
                        include_other_files,
                    )
                )
            else:
                logging.warning(f"No corresponding SRT file for {video_file}")

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error processing video: {e}")

    # Wait for all threads to complete
    for t in threadsList:
        t.join()


def process_video(
    video_file,
    srt_file,
    keywords,
    pre_duration,
    post_duration,
    threshold,
    output_folder,
    include_other_files,
):
    """
    Processes a video file by extracting clips around specified keywords found in the associated SRT file.
    Parameters:
    video_file (str): Path to the video file to be processed.
    srt_file (str): Path to the SRT file containing subtitles for the video.
    keywords (list of str): List of keywords to search for in the SRT file.
    pre_duration (float): Duration (in seconds) to include before the keyword timestamp in the clip.
    post_duration (float): Duration (in seconds) to include after the keyword timestamp in the clip.
    threshold (float): Confidence threshold for keyword detection.
    output_folder (str): Directory where the output clips and other files will be saved.
    include_other_files (bool): Whether to include additional files (frame, audio, transcript, metadata, text) in the output.
    Returns:
    None
    """
    try:
        logging.info(f"Processing video: {video_file} with SRT: {srt_file}")
        video_duration = get_video_duration(video_file)
        srt_duration = get_srt_duration(srt_file)

        if abs(video_duration - srt_duration) / video_duration > 0.01:
            e = f"Duration mismatch: {video_file} (video: {video_duration}s, srt: {srt_duration}s). Skipping."
            logging.error(e)
            print(e)
            return

        for keyword in keywords:
            results = find_keyword_timestamps(srt_file, keyword, threshold)
            if not results:
                i = f"Keyword '{keyword}' not found in {srt_file}."
                logging.info(i)
                print(i)
                continue

            clip_ranges = []
            for timestamp, _, confidence in results:
                timestamp_seconds = timestamp_to_seconds(timestamp)
                start_time = max(0, timestamp_seconds - pre_duration)
                end_time = timestamp_seconds + post_duration

                if any(start <= timestamp_seconds <= end for start, end in clip_ranges):
                    logging.info(
                        f"Timestamp {timestamp} for keyword '{keyword}' is within an existing clip range. Skipping."
                    )
                    continue

                clip_ranges.append((start_time, end_time))
                logging.info(
                    f"Processing keyword '{keyword}' at timestamp {timestamp} (start_time: {start_time}, end_time: {end_time})"
                )

                keyword_output_folder = os.path.join(output_folder, keyword)
                if not os.path.exists(keyword_output_folder):
                    os.makedirs(keyword_output_folder)

                sanitized_filename = sanitize_filename(os.path.basename(video_file))
                output_base = os.path.join(
                    keyword_output_folder, f"{sanitized_filename}_{timestamp}"
                )

                video_output_file = f"{output_base}.mp4"
                if not os.path.exists(video_output_file):
                    clip_video(
                        video_file,
                        timestamp_seconds,
                        timestamp_seconds,
                        pre_duration,
                        post_duration,
                        video_output_file,
                    )

                if include_other_files:
                    frame_output_file = f"{output_base}.png"
                    audio_output_file = f"{output_base}.mp3"
                    transcript_output_file = f"{output_base}.srt"
                    metadata_output_file = f"{output_base}.json"
                    txt_output_file = f"{output_base}.txt"

                    if not os.path.exists(frame_output_file):
                        extract_frame(video_file, timestamp_seconds, frame_output_file)
                    if not os.path.exists(audio_output_file):
                        extract_audio(
                            video_file,
                            start_time,
                            end_time - start_time,
                            audio_output_file,
                        )
                    if not os.path.exists(transcript_output_file):
                        extract_transcript(
                            srt_file,
                            start_time,
                            end_time - start_time,
                            transcript_output_file,
                        )

                    if os.path.exists(metadata_output_file):
                        logging.info(
                            f"Metadata file {metadata_output_file} already exists. Skipping."
                        )
                    else:
                        # Write metadata
                        metadata = {
                            "original_file": video_file,
                            "timestamp": timestamp,
                            "start_time": start_time,
                            "end_time": end_time,
                            "confidence": confidence,
                        }
                    with open(metadata_output_file, "w") as json_file:
                        json.dump(metadata, json_file)

                    if os.path.exists(txt_output_file):
                        logging.info(
                            f"Text file {txt_output_file} already exists. Skipping."
                        )
                    else:
                        # Write center point, keyword, and confidence to .txt file
                        with open(txt_output_file, "w") as txt_file:
                            txt_file.write(f"Timestamp: {timestamp}\n")
                            txt_file.write(f"Keyword: {keyword}\n")
                            txt_file.write(f"Confidence: {confidence}\n")

    except Exception as e:
        logging.error(f"Error processing {video_file} with keyword '{keyword}': {e}")


def main():
    """
    Parses command-line arguments and initiates the video processing.
    """
    parser = argparse.ArgumentParser(
        description="Clip videos based on keywords in transcript."
    )
    parser.add_argument(
        "-s",
        "--source-folder",
        default="./source",
        help="The folder containing the video files and accompanying .srt files.",
    )
    parser.add_argument(
        "-o",
        "--output-folder",
        default="./output",
        help="The folder to save the clipped videos and extracted frames.",
    )
    parser.add_argument(
        "-k",
        "--keywords",
        type=str,
        required=True,
        help="Comma-separated keywords to search for in the transcript.",
    )
    parser.add_argument(
        "-p",
        "--pre-duration",
        type=int,
        default=5,
        help="The duration (in seconds) to include before the keyword timestamp.",
    )
    parser.add_argument(
        "-d",
        "--post-duration",
        type=int,
        default=30,
        help="The duration (in seconds) to include after the keyword timestamp.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=int,
        default=95,
        help="The fuzzy matching threshold for keyword matching.",
    )
    parser.add_argument(
        "-i",
        "--include-other-files",
        action="store_true",
        help="Include other files (frames, audio, transcript, metadata) in the output. Default is to include only the .mp4 video clip.",
    )
    args = parser.parse_args()

    keywords = args.keywords.split(",")

    process_videos_parallel(
        args.source_folder,
        args.output_folder,
        keywords,
        args.pre_duration,
        args.post_duration,
        args.threshold,
        args.include_other_files,
    )


if __name__ == "__main__":
    main()
