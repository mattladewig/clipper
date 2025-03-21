import argparse
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from transformers import pipeline
import shutil
import uuid
import srt
from tqdm import tqdm
from subtitle_parser import find_keywords, get_all_search_targets, load_subtitles_stream
from config import ClipperConfig
import keyboard
import time
import sys
import signal
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global state for interactive control
pause_event = threading.Event()
exit_event = threading.Event()
temp_files_lock = threading.Lock()
temp_files = []

def signal_handler(sig, frame):
    logger.info("Caught Ctrl+C, shutting down...")
    exit_event.set()
    cleanup_temp_files()
    sys.exit(0)

def cleanup_temp_files():
    global temp_files
    with temp_files_lock:
        for temp_file in temp_files[:]:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    logger.debug(f"Deleted temp file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_file}: {e}")
        temp_files.clear()

def keyboard_listener():
    while not exit_event.is_set():
        if keyboard.is_pressed('q'):
            logger.info("'q' pressed, exiting...")
            exit_event.set()
            cleanup_temp_files()
            sys.exit(0)
        if keyboard.is_pressed('p') and not pause_event.is_set():
            logger.info("Pausing processing...")
            pause_event.set()
        if keyboard.is_pressed('r') and pause_event.is_set():
            logger.info("Resuming processing...")
            pause_event.clear()
        time.sleep(0.1)

def sanitize_filename(filename: str, max_length: int = 200) -> str:
    invalid_chars = r'[<>:"/\\|?*\0]'
    safe_name = re.sub(invalid_chars, "_", filename)
    safe_name = re.sub(r"_+", "_", safe_name)
    return safe_name[:max_length].strip("_")

def merge_subtitle_ranges(
    matched_subtitles: List[srt.Subtitle], pre_buffer: float, post_buffer: float
) -> List[Tuple[float, float, List[srt.Subtitle]]]:
    if not matched_subtitles:
        return []

    matched_subtitles.sort(key=lambda x: x.start.total_seconds())
    merged_clips = []
    current_subs = [matched_subtitles[0]]
    current_start = max(0, matched_subtitles[0].start.total_seconds() - pre_buffer)
    current_end = matched_subtitles[0].end.total_seconds() + post_buffer

    for sub in matched_subtitles[1:]:
        start = sub.start.total_seconds()
        end = sub.end.total_seconds()
        buffered_start = max(0, start - pre_buffer)
        buffered_end = end + post_buffer

        if buffered_start <= current_end:
            current_subs.append(sub)
            current_end = max(current_end, buffered_end)
        else:
            merged_clips.append((current_start, current_end, current_subs))
            current_start = buffered_start
            current_end = buffered_end
            current_subs = [sub]

    merged_clips.append((current_start, current_end, current_subs))
    return merged_clips

def process_video(
    video_file: str,
    subtitle_file: str,
    output_dir: str,
    keywords: List[str],
    word_alt_map: Dict[str, List[str]],
    pre_buffer: float,
    post_buffer: float,
    use_subdirs: bool,
    speech_categories: List[str] = None,
) -> None:
    if exit_event.is_set():
        return

    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg not found in PATH.")

    # Use facebook/bart-large-mnli for better zero-shot performance
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)

    project_root = Path(".")
    video_path = Path(video_file)
    video_id = sanitize_filename(video_path.stem)
    output_subdir = project_root / output_dir / video_id if use_subdirs else project_root / output_dir
    output_subdir.mkdir(parents=True, exist_ok=True)

    tmp_dir = project_root / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    tmp_video_name = f"temp_input_{uuid.uuid4().hex}_{sanitize_filename(video_path.name)}"
    tmp_video_path = tmp_dir / tmp_video_name
    shutil.copy2(video_path, tmp_video_path)
    tmp_video_rel_path = tmp_video_path.relative_to(project_root).as_posix()
    with temp_files_lock:
        temp_files.append(tmp_video_path)

    subtitles = list(load_subtitles_stream(subtitle_file))
    search_targets, _ = get_all_search_targets(set(keywords), word_alt_map)
    logger.debug(f"Search targets: {search_targets}")
    matched_subtitles = find_keywords(subtitles, search_targets) if keywords else subtitles

    if not matched_subtitles:
        logger.info(f"No search targets found in {subtitle_file}")
        with temp_files_lock:
            if tmp_video_path in temp_files and tmp_video_path.exists():
                tmp_video_path.unlink()
                temp_files.remove(tmp_video_path)
        return

    if speech_categories:
        categorized_subtitles = []
        for i, sub in enumerate(subtitles):
            if exit_event.is_set():
                cleanup_temp_files()
                return
            while pause_event.is_set() and not exit_event.is_set():
                time.sleep(0.1)
            ##TODO add config file var for look-ahead window size
            start_idx = max(0, i - 2)
            ##TODO add config file var for look-behind window size
            end_idx = min(len(subtitles), i + 3)
            context = " ".join(s.content for s in subtitles[start_idx:end_idx])
            result = classifier(context, speech_categories, multi_label=True)
            scores = result["scores"]
            labels = result["labels"]
            ##TODO add config var for threshold of the score
            detected_categories = [label for label, score in zip(labels, scores) if score > 0.9]
            if detected_categories:
                sub.categories = detected_categories
                categorized_subtitles.append(sub)
            logger.debug(f"Subtitle: '{sub.content}' | Context: '{context}' | Categories: {detected_categories}")
            logger.debug(f"Scores: {dict(zip(labels, scores))}")
        matched_subtitles = categorized_subtitles if categorized_subtitles else matched_subtitles

    clips = merge_subtitle_ranges(matched_subtitles, pre_buffer, post_buffer)

    for idx, (start_time, end_time, clip_subs) in enumerate(clips, 1):
        if exit_event.is_set():
            cleanup_temp_files()
            return
        while pause_event.is_set() and not exit_event.is_set():
            time.sleep(0.1)

        duration = end_time - start_time
        if start_time < 0:
            start_time = 0
            duration = end_time

        clip_subs_filtered = [
            sub for sub in subtitles
            if sub.start.total_seconds() >= start_time and sub.end.total_seconds() <= end_time
        ]
        logger.debug(f"Filtered subtitles for clip {idx}: {[sub.content for sub in clip_subs_filtered]}")

        for sub in clip_subs_filtered:
            sub.start = sub.start - timedelta(seconds=start_time)
            sub.end = sub.end - timedelta(seconds=start_time)

        tmp_srt_path = tmp_dir / f"temp_{uuid.uuid4().hex}.srt"
        with open(tmp_srt_path, mode="w", encoding='utf-8') as tmp_srt:
            tmp_srt.write(srt.compose(clip_subs_filtered))
        with temp_files_lock:
            temp_files.append(tmp_srt_path)
        tmp_srt_rel_path = tmp_srt_path.relative_to(project_root).as_posix()

        if speech_categories and any(hasattr(sub, "categories") for sub in clip_subs_filtered):
            matched_categories = sorted(set(
                cat for sub in clip_subs_filtered if hasattr(sub, "categories") 
                for cat in sub.categories
            ))
            categories_str = "_".join(matched_categories) if matched_categories else "uncategorized"
        else:
            matched_keywords = sorted(set(
                kw for kw in search_targets 
                if any(kw in sub.content.lower() for sub in clip_subs_filtered)
            ))
            categories_str = "_".join(matched_keywords) if matched_keywords else "unknown"

        base_output_name = f"{video_id}-{categories_str}_{idx:03d}_{int(start_time)}"
        safe_output_name = sanitize_filename(base_output_name)
        output_file = output_subdir / f"{safe_output_name}.mp4"
        output_file_rel = output_file.relative_to(project_root).as_posix()
        logger.debug(f"Temporary SRT for {output_file_rel}:\n{srt.compose(clip_subs_filtered)}")
        logger.debug(f"Output file path: {output_file_rel}")

        ffmpeg_cmd = (
            f'ffmpeg -ss {start_time} -t {duration} -i "{tmp_video_rel_path}" '
            f'-vf "scale=-2:720" '
            f'-vf "subtitles=\'{tmp_srt_rel_path}\'" '
            f'-c:v libx264 -c:a aac -y "{output_file_rel}"'
        )
        logger.debug(f"FFmpeg command: {ffmpeg_cmd}")

        try:
            result = subprocess.run(
                ffmpeg_cmd,
                shell=True,
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            logger.debug(f"FFmpeg stdout: {result.stdout}")
            logger.info(f"Clipped video with embedded subtitles saved to {output_file_rel}")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed with exit code {e.returncode}: {e.stderr}")
            raise

    if not exit_event.is_set():
        cleanup_temp_files()

def main():
    parser = argparse.ArgumentParser(description="Clip videos based on subtitle keywords.")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON config file")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    config = ClipperConfig.from_file(Path(args.config))
    logging_level = logging.DEBUG if args.verbose else getattr(logging, config.logging)
    logger.setLevel(logging_level)

    signal.signal(signal.SIGINT, signal_handler)
    listener_thread = threading.Thread(target=keyboard_listener, daemon=True)
    listener_thread.start()

    input_dir = config.directory or "videos"
    output_dir = config.output_dir
    keywords = config.keywords
    speech_categories = config.speech_categories
    word_alt_map = config.word_alt_map or {}
    pre_buffer = config.pre_buffer if config.pre_buffer is not None else 5.0
    post_buffer = config.post_buffer if config.post_buffer is not None else 5.0
    max_workers = config.max_workers if config.max_workers is not None else 1
    use_subdirs = config.use_subdirs

    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".mp4")]
    if not video_files:
        logger.warning(f"No .mp4 files found in {input_dir}")
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for video_file in video_files:
            if exit_event.is_set():
                break
            video_path = os.path.join(input_dir, video_file)
            subtitle_file = os.path.join(input_dir, f"{Path(video_file).stem}.srt")
            if not os.path.exists(subtitle_file):
                logger.warning(f"Subtitle file not found for {video_file}, skipping")
                continue
            logger.info(f"Processing {video_file}")
            futures.append(
                executor.submit(
                    process_video,
                    video_path,
                    subtitle_file,
                    output_dir,
                    keywords,
                    word_alt_map,
                    pre_buffer,
                    post_buffer,
                    use_subdirs,
                    speech_categories,
                )
            )
        for future in tqdm(futures, desc="Processing videos"):
            if exit_event.is_set():
                break
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error processing video: {e}")
    
    if not exit_event.is_set():
        cleanup_temp_files()

if __name__ == "__main__":
    main()