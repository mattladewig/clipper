import argparse
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict
from datetime import timedelta

import srt
from tqdm import tqdm
from subtitle_parser import find_keywords, get_all_search_targets, load_subtitles_stream

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def merge_subtitle_ranges(
    matched_subtitles: List[srt.Subtitle], pre_buffer: float, post_buffer: float
) -> List[Tuple[float, float, List[srt.Subtitle]]]:
    """Merge overlapping subtitle ranges with consistent buffers."""
    if not matched_subtitles:
        return []

    # Sort subtitles by start time
    matched_subtitles.sort(key=lambda x: x.start.total_seconds())

    # Initialize merged clips
    merged_clips = []
    current_subs = [matched_subtitles[0]]
    current_start = max(0, matched_subtitles[0].start.total_seconds() - pre_buffer)
    current_end = matched_subtitles[0].end.total_seconds() + post_buffer
    logger.debug(f"Initial: start={current_start}, end={current_end}, sub={matched_subtitles[0].content}")

    for sub in matched_subtitles[1:]:
        start = sub.start.total_seconds()
        end = sub.end.total_seconds()
        buffered_start = max(0, start - pre_buffer)
        buffered_end = end + post_buffer
        logger.debug(f"Processing: sub={sub.content}, start={start}, end={end}, buffered_start={buffered_start}, buffered_end={buffered_end}")

        if buffered_start <= current_end:  # Overlap with current clip
            current_subs.append(sub)
            current_end = max(current_end, buffered_end)
            logger.debug(f"Overlap detected: extending end to {current_end}")
        else:
            merged_clips.append((current_start, current_end, current_subs))
            logger.debug(f"New clip added: {current_start} to {current_end}, subs={[s.content for s in current_subs]}")
            current_start = buffered_start
            current_end = buffered_end
            current_subs = [sub]
            logger.debug(f"New range started: start={current_start}, end={current_end}")

    merged_clips.append((current_start, current_end, current_subs))
    logger.debug(
        f"Final clips (pre={pre_buffer}s, post={post_buffer}s): {[(s, e, [sub.content for sub in subs]) for s, e, subs in merged_clips]}"
    )
    return merged_clips

def process_video(
    video_file: str,
    subtitle_file: str,
    output_dir: str,
    keywords: List[str],
    word_alt_map: Dict[str, List[str]],
    pre_buffer: float,
    post_buffer: float,
) -> None:
    """Process a video file, extract clips based on keywords, and save with subtitles."""
    import tempfile

    video_id = Path(video_file).stem
    output_subdir = os.path.join(output_dir, video_id)
    os.makedirs(output_subdir, exist_ok=True)

    # Load subtitles
    subtitles = list(load_subtitles_stream(subtitle_file))

    # Get search targets
    search_targets, _ = get_all_search_targets(set(keywords), word_alt_map)
    logger.debug(f"Search targets: {search_targets}")
    matched_subtitles = find_keywords(subtitles, search_targets)

    if not matched_subtitles:
        logger.info(f"No search targets found in {subtitle_file}")
        return

    # Merge subtitle ranges into non-overlapping clips
    clips = merge_subtitle_ranges(matched_subtitles, pre_buffer, post_buffer)

    for idx, (start_time, end_time, clip_subs) in enumerate(clips, 1):
        duration = end_time - start_time
        if start_time < 0:
            start_time = 0  # Adjust negative start times
            duration = end_time

        # Filter subtitles for this clip range
        clip_subs_filtered = [
            sub for sub in subtitles
            if sub.start.total_seconds() >= start_time and sub.end.total_seconds() <= end_time
        ]
        logger.debug(f"Filtered subtitles for clip {idx}: {[sub.content for sub in clip_subs_filtered]}")

        # Adjust subtitle timings relative to clip start
        for sub in clip_subs_filtered:
            sub.start = sub.start - timedelta(seconds=start_time)
            sub.end = sub.end - timedelta(seconds=start_time)

        # Write temporary SRT file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as tmp_srt:
            tmp_srt.write(srt.compose(clip_subs_filtered))
            tmp_srt_path = tmp_srt.name

        logger.debug(f"Temporary SRT for {os.path.join(output_subdir, f'{video_id}-{search_targets[0]}_{idx:03d}_{int(start_time)}.mp4')}:\n{srt.compose(clip_subs_filtered)}")

        # Define output file path
        matched_keyword = next(kw for kw in search_targets if any(kw in sub.content.lower() for sub in clip_subs))
        output_file = os.path.join(output_subdir, f"{video_id}-{matched_keyword}_{idx:03d}_{int(start_time)}.mp4")
        #output_file = os.path.join(output_subdir, f"{video_id}-{search_targets[0]}_{idx:03d}_{int(start_time)}.mp4")
        logger.debug(f"Output file path: {output_file}")

        # FFmpeg command for video with embedded subtitles
        ffmpeg_cmd = [
            "ffmpeg",
            "-ss", str(start_time),
            "-t", str(duration),
            "-i", video_file,
            "-vf", f"subtitles={tmp_srt_path}",
            "-c:v", "libx264",
            "-c:a", "aac",
            "-y",  # Overwrite output file if it exists
            output_file
        ]
        logger.debug(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")

        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
            logger.info(f"Clipped video with embedded subtitles saved to {output_file}")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed with exit code {e.returncode}: {e.stderr}")
            raise
        finally:
            os.unlink(tmp_srt_path)  # Clean up temporary SRT file

def main():
    """Main function to parse arguments and process videos."""
    parser = argparse.ArgumentParser(description="Clip videos based on subtitle keywords.")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON config file")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose or (hasattr(args, 'config') and json.load(open(args.config)).get("verbose", False)):
        logger.setLevel(logging.DEBUG)

    # Load config
    with open(args.config, "r") as f:
        config = json.load(f)

    input_dir = config.get("directory", "videos")
    output_dir = config.get("output_dir", "clips")
    keywords = config.get("keywords", [])
    word_alt_map = config.get("word_alt_map", {})
    pre_buffer = config.get("pre_buffer", 5.0)
    post_buffer = config.get("post_buffer", 5.0)

    # Find all video files
    video_files = [f for f in os.listdir(input_dir) if f.endswith(".mp4")]
    if not video_files:
        logger.warning(f"No .mp4 files found in {input_dir}")
        return

    # Process each video
    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(input_dir, video_file)
        subtitle_file = os.path.join(input_dir, f"{Path(video_file).stem}.srt")

        if not os.path.exists(subtitle_file):
            logger.warning(f"Subtitle file not found for {video_file}, skipping")
            continue

        logger.info(f"Processing {video_file}")
        process_video(
            video_path,
            subtitle_file,
            output_dir,
            keywords,
            word_alt_map,
            pre_buffer,
            post_buffer,
        )

if __name__ == "__main__":
    main()