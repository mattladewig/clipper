import os
import re
import logging
import argparse
import threading
import time
from datetime import timedelta
from typing import Any, List, LiteralString
from fuzzywuzzy import fuzz
import ffmpeg
from ffmpeg import Error as FFMpegError
import nltk
from nltk.stem import WordNetLemmatizer as wnl
from nltk.corpus import wordnet
from word_alt_map import word_alt_map
from collections import defaultdict
import tempfile


exit_flag = threading.Event()


class ProcessMetaThread(threading.Thread):
    """
    A thread class that processes metadata in the background.
    This thread runs as a daemon and continuously lemmatizes the word "dog"
    until an exit flag is set.
    Methods:
    --------
    __init__():
        Initializes the thread and sets it as a daemon.
    run():
        The main logic of the thread. It lemmatizes the word "dog" in a loop
        with a small sleep interval to prevent busy-waiting.
    """

    def __init__(self):
        threading.Thread.__init__(self)
        self.daemon = True  # Set the daemon attribute directly

    def run(self):
        lemmatizer = wnl()
        while not exit_flag.is_set():
            lemmatizer.lemmatize("dog")
            # Add a small sleep to prevent busy-waiting
            time.sleep(0.1)


def generate_bidirectional_mapping(base_dict):
    """
    Generates a bidirectional mapping from the given dictionary. For each key-value pair in the
    input dictionary, it creates a mapping in both directions, ensuring that each key and value
    can be accessed from one another.

    Args:
        base_dict (dict): A dictionary where each key maps to a list of values.

    Returns:
        dict: A dictionary where each key maps to a list of values, including the bidirectional
        mappings.
    """
    bidirectional_dict = {}
    for key, values in base_dict.items():
        if key not in bidirectional_dict:
            bidirectional_dict[key] = set(values)
        for value in values:
            if value not in bidirectional_dict:
                bidirectional_dict[value] = set()
            bidirectional_dict[value].add(key)
            bidirectional_dict[value].update(v for v in values if v != value)
    # Convert sets back to lists and sort them
    for key in bidirectional_dict:
        bidirectional_dict[key] = sorted(list(bidirectional_dict[key]))
    return bidirectional_dict


def configure_logging(args):
    """
    Configures the logging settings for the application.

    Parameters:
    args (Namespace): A namespace object containing the command-line arguments.
                      It should have the attributes 'debug' and 'verbose' to
                      determine the logging level.

    The function creates a log directory if it doesn't exist and configures
    logging to write to a file named 'clipper.log' in the log directory.
    The logging level is set based on the 'debug' and 'verbose' attributes
    of the args parameter:
        - If 'debug' is True, the logging level is set to DEBUG.
        - If 'verbose' is True, the logging level is set to INFO.
        - Otherwise, the logging level is set to WARNING.
    """

    # Create log directory if it doesn't exist
    log_directory = "./log"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # Configure logging to write to a file in the log directory
    log_file: LiteralString = os.path.join(log_directory, "clipper.log")

    # Configure logging
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filename=log_file,
            filemode="a",
        )
    elif args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filename=log_file,
            filemode="a",
        )
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(console_handler)
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filename=log_file,
            filemode="a",
        )


def get_all_search_targets(keywords):
    """
    Generates a comprehensive list of search targets based on the provided keywords.
    This function takes a list of keywords and expands it by including various forms
    and alternatives of each keyword. It uses a lemmatizer to generate base forms of
    the words and applies rules to generate present participle, past tense, and plural
    forms where applicable. Custom word mappings are also considered.
    Args:
        keywords (list of str): A list of keywords to generate search targets from.
    Returns:
        list of str: A list of expanded search targets including various forms and
                    alternatives of the provided keywords.
    Logs:
        The function logs various stages of the process including initial keywords,
        intermediate search targets, identified word types, and final search targets.
    """
    log = f"Initial search targets: {keywords}"
    logging.info(log)

    all_search_targets = []
    search_targets = []
    lemmatizer = wnl()

    # Generate the bidirectional mapping
    custom_alternatives = generate_bidirectional_mapping(word_alt_map)

    # Add common alternatives
    for k in keywords:
        search_targets.append(k)
        if k in custom_alternatives:
            log = f"Found custom word map alternatives for '{k}': {custom_alternatives[k]}"
            logging.info(log)
            search_targets.extend(custom_alternatives[k])

    log: str = f"Intermediate search targets: {search_targets}"
    logging.info(log)

    for t in search_targets:
        # Initialize lists for different forms of the word
        present_participle = []
        past_tense = []
        plural_noun = []
        irregular_plurals = dict()
        irregular_plural_noun = []
        word_type = []  # Other

        # Determine the type of the word
        if wordnet.synsets(t, pos=wordnet.VERB):
            word_type.extend("v")
        if wordnet.synsets(t, pos=wordnet.NOUN):
            word_type.extend("n")

        # Log the word type
        log = f"Word '{t}' is identified as a {word_type}."
        logging.info(log)
        # Generate present participle and past tense forms for verbs

        if "v" in word_type:
            lem = lemmatizer.lemmatize(t, "v")
            log = f"lem: {lem}"
            logging.info(log)

            all_search_targets.append(lem)

            # past_tense verb rules:

            ## Ends in "y": Change the "y" to "i" and add "-ed", e.g. "study" -> "studied"
            if lem[-1] in "y" and lem[-2] not in "aeiou":
                past_tense: List[str] = [lem[:-1] + "ied"]

            # rules for present_participle:
            ## Verbs ending in "-e": Drop the "-e" and add "-ing". For example, "make" becomes "making".
            ## Verbs ending in "-ie": Drop the "-ie", add a "-y", and then add "-ing". For example, "lie" becomes "lying".
            if lem[-1] in "e" and lem[-2] not in "ei":
                present_participle = [lem[:-1] + "ing"]
            elif lem.endswith("ie"):
                present_participle: List[str] = [lem[:-2] + "ying"]

            if present_participle:
                all_search_targets.extend(present_participle)
            if past_tense:
                all_search_targets.extend(past_tense)

        if "n" in word_type:
            lem: str = lemmatizer.lemmatize(t, "n")
            log = f"lem: {lem}"
            logging.info(log)
            all_search_targets.append(lem)

            # Add irregular plural forms for common singular nouns
            irregular_plurals: dict = {
                "child": "children",
                "man": "men",
                "woman": "women",
                "tooth": "teeth",
                "foot": "feet",
                "mouse": "mice",
                "goose": "geese",
                "louse": "lice",
                "die": "dice",
                "person": "people",
                "cactus": "cacti",
                "focus": "foci",
                "fungus": "fungi",
                "nucleus": "nuclei",
                "syllabus": "syllabi",
                "analysis": "analyses",
                "diagnosis": "diagnoses",
                "oasis": "oases",
                "thesis": "theses",
                "crisis": "crises",
                "phenomenon": "phenomena",
                "criterion": "criteria",
                "datum": "data",
            }

            if lem in irregular_plurals:
                irregular_plural_noun = [irregular_plurals[lem]]
                all_search_targets.extend(irregular_plural_noun)
            # Handle reverse mapping for irregular plurals
            reverse_irregular_plurals = {v: k for k, v in irregular_plurals.items()}
            if lem in reverse_irregular_plurals:
                irregular_plural_noun = [reverse_irregular_plurals[lem]]
                all_search_targets.extend(irregular_plural_noun)

            ## If the word ends in "-y" and the letter before the "-y" is a consonant, change the "-y" to "-ies".
            if lem.endswith("y") and lem[-2] not in "aeiou":
                plural_noun: List[Any] = [lem[:-1] + "ies"]
                all_search_targets.extend(plural_noun)

    all_search_targets.extend(search_targets)

    all_search_targets = [
        word for word in all_search_targets if len(word) > 2
    ]  # remove short words

    all_search_targets = sorted(
        list(set(all_search_targets))
    )  # final remove duplicates and sort

    log = f"Final search targets: {all_search_targets}"
    logging.info(log)

    return all_search_targets


def sanitize_filename(filename):
    """
    Sanitizes a given filename by removing non-ASCII characters and replacing spaces and special characters with underscores.
    Args:
        filename (str): The original filename to be sanitized.
    Returns:
        str: The sanitized filename with non-ASCII characters replaced by hyphens and spaces and special characters replaced by underscores.
    """

    sanitized: str = re.sub(r"[^\x00-\x7F]+", "-", filename)
    sanitized = re.sub(r"[ /:*?\"<>|\\]", "_", sanitized)
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


def clip_video(
    video_file,
    transcript_text,
    timestamp_start,
    timestamp_end,
    pre_duration,
    post_duration,
    output_folder,
    output_file,
):
    """
    Clips a segment from a video file and adds metadata to the output file.

    Parameters:
    video_file (str): Path to the input video file.
    transcript_text (str): Transcript text to be added as metadata.
    timestamp_start (str): Start timestamp of the clip in HH:MM:SS format.
    timestamp_end (str): End timestamp of the clip in HH:MM:SS format.
    pre_duration (int): Duration in seconds to include before the start timestamp.
    post_duration (int): Duration in seconds to include after the end timestamp.
    output_folder (str): Path to the folder where the output file will be saved.
    output_file (str): Name of the output video file.

    Raises:
    ffmpeg.Error: If an error occurs during the ffmpeg process.
    """
    video_duration: int = get_video_duration(video_file)
    start_time: int = max(0, timestamp_to_seconds(timestamp_start) - pre_duration)
    end_time: int = min(
        video_duration, timestamp_to_seconds(timestamp_end) + post_duration
    )
    log = f"Clipping video from {start_time} to {end_time}: {transcript_text}"
    logging.info(log)
    output_path = os.path.join(output_folder, output_file)
    output_path_sanitized = os.path.join(output_folder, sanitize_filename(output_file))
    # Check if the output file already exists
    if os.path.exists(output_path_sanitized):
        logging.info(
            f"Output file already exists: {output_path}. Skipping clip creation."
        )
        return
    srt_file = video_file.replace(".mp4", ".srt")
    srt_clip = clip_srt(srt_file, start_time, end_time)
    # Clip the video and add metadata using ffmpeg
    # Extract metadata from the source video file
    metadata = {}
    # Add additional metadata
    metadata["title"] = (
        f'"{transcript_text}"'
        + ", start:"
        + str(timestamp_start)
        + ", source:"
        + video_file
    )

    # Filter out invalid metadata entries
    valid_metadata = {
        k: v for k, v in metadata.items() if isinstance(k, str) and isinstance(v, str)
    }

    # Construct the ffmpeg command with metadata
    ffmpeg_input = ffmpeg.input(video_file, ss=start_time, to=end_time)
    ffmpeg_output = ffmpeg_input.output(
        output_path,
        c="copy",
        map_metadata=-1,
        # trunk-ignore(ruff/B035)
        **{"metadata": f"{k}={v}" for k, v in valid_metadata.items()},
    )

    # Log the ffmpeg command
    ffmpeg_command = ffmpeg_output.compile()
    logging.debug(f"Running ffmpeg command: {ffmpeg_command}")

    # Suppress console output unless debug is enabled
    try:
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            ffmpeg_output.run(overwrite_output=True)
        else:
            ffmpeg_output.run(quiet=True, overwrite_output=True)
    except FFMpegError as e:
        logging.error(f"ffmpeg error: {e}")
        if e.stderr:
            logging.error(f"ffmpeg stderr: {e.stderr.decode('utf-8')}")
        else:
            logging.error("ffmpeg error occurred, but no stderr output is available.")

    output_file_with_text = output_file.replace(".mp4", "_with_srt.mp4")
    # Add transcript text to the video
    add_srt_to_video(output_file, srt_clip, output_folder, output_file_with_text)


# create function to clip srt file for same start_time and end_time as video clip
def clip_srt(srt_file, start_time, end_time):
    """
    Clips a segment from an SRT file based on the specified start and end times and returns it as a string.
    Args:
        srt_file (str): Path to the input SRT file.
        start_time (int): Start time in seconds.
        end_time (int): End time in seconds.
    Returns:
        str: The clipped SRT content.
    Raises:
        ValueError: If the start_time is greater than the end_time.
    """
    if start_time > end_time:
        raise ValueError("start_time cannot be greater than end_time")

    with open(srt_file, "r") as file:
        content = file.read()

    matches = re.findall(
        r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*)", content
    )

    clipped_content = []
    sequence_number = 1
    for match in matches:
        timestamp_start = timestamp_to_seconds(match[0].split(",")[0])
        timestamp_end = timestamp_to_seconds(match[1].split(",")[0])
        if (
            start_time <= timestamp_start <= end_time
            or start_time <= timestamp_end <= end_time
        ):
            clipped_content.append(
                f"{sequence_number}\n{match[0]} --> {match[1]}\n{match[2]}\n"
            )
            sequence_number += 1

    return "\n".join(clipped_content)


def validate_srt_content(srt_content):
    """
    Validates the format of the SRT content.
    Args:
        srt_content (str): The SRT content to validate.
    Returns:
        bool: True if the SRT content is valid, False otherwise.
    """
    lines = srt_content.splitlines()
    for i, line in enumerate(lines):
        if i % 4 == 0:  # Sequence number
            if not line.isdigit():
                return False
        elif i % 4 == 1:  # Timestamp
            if not ("-->" in line and len(line.split("-->")) == 2):
                return False
        # Other lines can be empty or contain text
    return True


def add_srt_to_video(video_clip, srt_clip, output_folder, output_file):
    """
    Adds an SRT subtitle track to a video clip and saves the result as a new video file.
    Args:
        video_clip (str): Path to the input video clip.
        srt_clip (str): SRT content to be added as a subtitle track.
        output_file (str): Path to the output video file.
    Returns:
        None
    Raises:
        ffmpeg.Error: If there is an error during the ffmpeg command execution.
    """
    log = f"Adding SRT to video: {video_clip}, exporting new video to {output_file}"
    logging.info(log)

    # Validate SRT content
    if not validate_srt_content(srt_clip):
        raise ValueError("Invalid SRT content format")

    # Ensure the "tmp" directory exists
    tmp_dir = "tmp"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    # Create a temporary SRT file in the "tmp" directory
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".srt", dir=tmp_dir
    ) as temp_srt_file:
        temp_srt_file.write(srt_clip.encode("utf-8"))
        temp_srt_path = temp_srt_file.name

    # Ensure the temporary SRT file is written correctly
    if not os.path.exists(temp_srt_path):
        raise FileNotFoundError(f"Temporary SRT file not found: {temp_srt_path}")
    logging.debug(f"Temporary SRT file created: {temp_srt_path}")

    # Read temp_srt file to debug log
    with open(temp_srt_path, "r") as file:
        temp_srt_content = file.read()
        logging.debug(f"Temporary SRT file content: {temp_srt_content}")

    # Construct the ffmpeg command to add the SRT subtitle track
    input_clip_path = os.path.join(output_folder, video_clip)
    output_clip_path = os.path.join(output_folder, sanitize_filename(output_file))
    try:
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            ffmpeg.input(input_clip_path).output(
                output_clip_path,
                vf=f"subtitles={temp_srt_path}",
                vcodec="libx264",
                acodec="copy",
            ).run(overwrite_output=True)
        else:
            ffmpeg.input(input_clip_path).output(
                output_clip_path,
                vf=f"subtitles={temp_srt_path}",
                vcodec="libx264",
                acodec="copy",
            ).run(quiet=True, overwrite_output=True)
    except ffmpeg.Error as e:
        logging.error(f"ffmpeg error: {e}")
        raise e

    # Clean up the temporary SRT file
    try:
        if os.path.exists(temp_srt_path):
            os.remove(temp_srt_path)
            logging.debug(f"Successfully removed temporary SRT file: {temp_srt_path}")
    except OSError as e:
        logging.error(f"Error removing temporary SRT file {temp_srt_path}: {e}")
    # Clean up the source clip
    try:
        if os.path.exists(input_clip_path):
            os.remove(input_clip_path)
            logging.debug(f"Successfully removed source clip: {input_clip_path}")
    except OSError as e:
        logging.error(f"Error removing source clip: {input_clip_path}: {e}")


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
    log = "Extracting frame: {video_file}, timestamp: {timestamp}, output_file: {output_file}"
    logging.info(log)

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


def find_keyword_timestamps(srt_file, all_search_targets, threshold):
    """
    Find timestamps in an SRT file where specified keywords appear with a similarity score above a given threshold.

    Args:
        srt_file (str): Path to the SRT file to be processed.
        all_search_targets (list of str): List of keywords to search for in the transcript.
        threshold (int): Minimum similarity score (0-100) for a word to be considered a match.

    Returns:
        list of tuples: A list of tuples containing:
            - srt_file (str): The path to the SRT file.
            - timestamp_start (str): The start timestamp of the subtitle segment.
            - timestamp_end (str): The end timestamp of the subtitle segment.
            - transcript_text (str): The full text of the subtitle segment.
            - keyword (str): The keyword that was matched.
            - matched_word (str): The word in the transcript that matched the keyword.
            - fuzz_score (int): The similarity score of the match.
    """
    with open(srt_file, "r") as file:
        content = file.read()

    # find single line of transcript text
    matches = re.finditer(
        r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*)", content
    )
    results = []

    for match in matches:
        timestamp_start = match.group(1).split(",")[0]  # Ignore milliseconds
        timestamp_end = match.group(2).split(",")[0]  # Ignore milliseconds
        transcript_text = match.group(3)
        transcript_words = transcript_text.split()
        for w in transcript_words:
            for keyword in all_search_targets:
                fuzz_score: int = fuzz.ratio(w.lower(), keyword.lower())
                if fuzz_score >= threshold:
                    results.append(
                        (
                            srt_file,
                            timestamp_start,
                            timestamp_end,
                            transcript_text,
                            keyword,
                            w,
                            fuzz_score,
                        )
                    )
                    break

    if results:
        logging.info(f"Results: {results}")
        return results
    else:
        logging.info("No results found.")
        return None


def get_video_duration(video_file):
    """
    Get the duration of a video file in seconds.
    Args:
        video_file (str): The path to the video file.
    Returns:
        int: The duration of the video in seconds.
    Raises:
        FFMpegError: If there is an error probing the video file.
    """
    try:
        probe = ffmpeg.probe(video_file)
        duration = float(probe["format"]["duration"])
        return int(duration)
    except FFMpegError as e:
        logging.error(f"Error probing video file {video_file}: {e}")
        raise e
    except KeyError:
        logging.error(f"Duration not found in probe result for {video_file}")
        raise KeyError("Duration not found in probe result") from None


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

    matches: list[Any] = re.findall(
        r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})", content
    )
    if not matches:
        return 0

    start_timestamp = matches[0][0].split(",")[0]  # Ignore milliseconds
    end_timestamp = matches[-1][1].split(",")[0]  # Ignore milliseconds

    start_seconds: int = timestamp_to_seconds(start_timestamp)
    end_seconds: int = timestamp_to_seconds(end_timestamp)

    return end_seconds - start_seconds


def process_all_videos(
    source_folder,
    output_folder,
    all_search_targets,
    pre_duration,
    post_duration,
    threshold,
    include_other_files,
):
    """
    Processes all video files in the specified source folder, searching for keywords in their corresponding SRT files,
    and clips segments of the videos based on the keyword timestamps.

    Args:
        source_folder (str): The folder containing the source video files.
        output_folder (str): The folder where the processed video clips will be saved.
        all_search_targets (list): A list of keywords to search for in the SRT files.
        pre_duration (int): The duration (in seconds) to include before the keyword timestamp in the clip.
        post_duration (int): The duration (in seconds) to include after the keyword timestamp in the clip.
        threshold (float): The threshold for keyword matching.
        include_other_files (bool): Whether to include other non-video files in the processing.

    Returns:
        dict: A dictionary with video file paths as keys and another dictionary as values,
              where the inner dictionary has keywords as keys and their counts as values.
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

    keyword_counts = defaultdict(lambda: defaultdict(int))

    for video_file in video_files:
        srt_file = video_file.replace(".mp4", ".srt")
        if os.path.exists(srt_file):
            results = find_keyword_timestamps(srt_file, all_search_targets, threshold)
            if results:
                processed_time_ranges = []
                for result in results:
                    keyword = result[4]
                    timestamp_start: int = timestamp_to_seconds(result[1])
                    timestamp_end: int = timestamp_to_seconds(result[2])

                    # Check if the current keyword instance falls within any existing clip time range
                    overlap = False
                    for start, end in processed_time_ranges:
                        if (
                            start <= timestamp_start <= end
                            or start <= timestamp_end <= end
                        ):
                            overlap = True
                            break

                    if overlap:
                        continue  # Skip if keyword instance is already included in a previous clip

                    # Add the new time range to the list
                    processed_time_ranges.append(
                        (timestamp_start - pre_duration, timestamp_end + post_duration)
                    )

                    transcript_text = result[3]
                    output_file: str = (
                        f"{os.path.splitext(os.path.basename(video_file))[0]}_{transcript_text}.mp4"
                    )
                    clip_video(
                        video_file,
                        transcript_text,
                        result[1],
                        result[2],
                        pre_duration,
                        post_duration,
                        output_folder,
                        output_file,
                    )
                    keyword_counts[video_file][keyword] += 1
        else:
            logging.info(f"No SRT file found for {video_file}")

    return keyword_counts


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
        help="Comma-separated keywords to search for in the transcript.",
    )
    parser.add_argument(
        "-f",
        "--file",
        help="Path to a file containing keywords, one per line.",
    )
    parser.add_argument(
        "-p",
        "--pre-duration",
        type=int,
        default=5,
        help="The duration (in seconds) to include before the keyword timestamp. Default 5 seconds.",
    )
    parser.add_argument(
        "-d",
        "--post-duration",
        type=int,
        default=10,
        help="The duration (in seconds) to include after the keyword timestamp. Default 10 seconds.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=int,
        default=99,
        help="The fuzzy matching threshold for keyword matching.",
    )
    parser.add_argument(
        "-i",
        "--include-other-files",
        action="store_true",
        help="Include other files (frames, audio, transcript, metadata) in the output. Default is to include only the .mp4 video clip.",
    )
    # New argument for logging level
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    args: argparse.Namespace = parser.parse_args()

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

    wordnet.ensure_loaded()

    configure_logging(args)

    # Determine the source of keywords
    if args.file and args.keywords:
        raise ValueError(
            "Both --keywords/-k and --file/-f cannot be provided together."
        )
    if args.file:
        with open(args.file, "r") as file:
            keywords: list[str] = [line.strip() for line in file if line.strip()]
    elif args.keywords:
        keywords = args.keywords.split(",")
    else:
        raise ValueError("Either --keywords/-k or --file/-f must be provided.")

    all_search_targets = get_all_search_targets(keywords)

    # Handle graceful exit on Ctrl+C
    try:
        if exit_flag.is_set():
            exit(1)
        keyword_counts = process_all_videos(
            args.source_folder,
            args.output_folder,
            all_search_targets,
            args.pre_duration,
            args.post_duration,
            args.threshold,
            args.include_other_files,
        )

        # Write summary to a file
        logging.debug(f"Keyword counts: {keyword_counts}")
        logging.debug(f"Result output to {all_search_targets}")
        if "keyword_counts" in locals():
            summary_file = os.path.join(args.output_folder, "keyword_summary.csv")
            logging.debug(f"Result output to {summary_file}")
            with open(summary_file, "w") as f:
                header: LiteralString = "video," + ",".join(all_search_targets)
                f.write(header + "\n")
                for video, counts in keyword_counts.items():
                    video: str = f'"{video}"'
                    row: list[str] = [video] + [
                        str(counts[t]) for t in all_search_targets
                    ]
                    f.write(",".join(row) + "\n")
            logging.info(f"Result output to {summary_file}")
            print(f"Summary of keyword results available at {summary_file}")
        else:
            logging.error("keyword_counts empty.")

    except KeyboardInterrupt:
        print("Exiting due to KeyboardInterrupt")


if __name__ == "__main__":
    main()
