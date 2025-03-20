import argparse
import logging
import os
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor

import srt
from tqdm import tqdm
from subtitle_parser import find_keywords, get_all_search_targets, load_subtitles_stream
from config import ClipperConfig

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

    matched_subtitles.sort(key=lambda x: x.start.total_seconds())
    merged_clips = []
    current_subs = [matched_subtitles[0]]
    current_start = max(0, matched_subtitles[0].start.total_seconds() - pre_buffer)
    current_end = matched_subtitles[0].end.total_seconds() + post_buffer
    logger.debug(
        f"Initial: start={current_start}, end={current_end}, sub={matched_subtitles[0].content}"
    )

    for sub in matched_subtitles[1:]:
        start = sub.start.total_seconds()
        end = sub.end.total_seconds()
        buffered_start = max(0, start - pre_buffer)
        buffered_end = end + post_buffer
        logger.debug(
            f"Processing: sub={sub.content}, start={start}, end={end}, buffered_start={buffered_start}, buffered_end={buffered_end}"
        )

        if buffered_start <= current_end:
            current_subs.append(sub)
            current_end = max(current_end, buffered_end)
            logger.debug(f"Overlap detected: extending end to {current_end}")
        else:
            merged_clips.append((current_start, current_end, current_subs))
            logger.debug(
                f"New clip added: {current_start} to {current_end}, subs={[s.content for s in current_subs]}"
            )
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
    use_subdirs: bool,
) -> None:
    """Process a video file, extract clips based on keywords, and save with subtitles at max 720p."""
    import tempfile

    video_id = Path(video_file).stem
    output_subdir = os.path.join(output_dir, video_id) if use_subdirs else output_dir
    os.makedirs(output_subdir, exist_ok=True)

    subtitles = list(load_subtitles_stream(subtitle_file))
    search_targets, _ = get_all_search_targets(set(keywords), word_alt_map)
    logger.debug(f"Search targets: {search_targets}")
    matched_subtitles = find_keywords(subtitles, search_targets)

    if not matched_subtitles:
        logger.info(f"No search targets found in {subtitle_file}")
        return

    clips = merge_subtitle_ranges(matched_subtitles, pre_buffer, post_buffer)

    # trunk-ignore(ruff/B007)
    for idx, (start_time, end_time, clip_subs) in enumerate(clips, 1):
        duration = end_time - start_time
        if start_time < 0:
            start_time = 0
            duration = end_time

        clip_subs_filtered = [
            sub
            for sub in subtitles
            if sub.start.total_seconds() >= start_time
            and sub.end.total_seconds() <= end_time
        ]
        logger.debug(
            f"Filtered subtitles for clip {idx}: {[sub.content for sub in clip_subs_filtered]}"
        )

        for sub in clip_subs_filtered:
            sub.start = sub.start - timedelta(seconds=start_time)
            sub.end = sub.end - timedelta(seconds=start_time)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".srt", delete=False
        ) as tmp_srt:
            tmp_srt.write(srt.compose(clip_subs_filtered))
            tmp_srt_path = tmp_srt.name

        # In process_video function, replace the matched_keywords block:
        matched_keywords = sorted(
            set(
                kw
                for kw in search_targets
                if any(
                    kw in sub.content.lower() for sub in clip_subs_filtered
                )  # Changed from clip_subs
            )
        )
        keywords_str = "_".join(matched_keywords) if matched_keywords else "unknown"
        output_file = os.path.join(
            output_subdir, f"{video_id}-{keywords_str}_{idx:03d}_{int(start_time)}.mp4"
        )
        logger.debug(
            f"Temporary SRT for {output_file}:\n{srt.compose(clip_subs_filtered)}"
        )
        logger.debug(f"Output file path: {output_file}")

        # FFmpeg command with explicit 404x720 scaling
        video_filters = [
            "scale=404:720:force_original_aspect_ratio=decrease",
            f"subtitles={tmp_srt_path}",
        ]
        ffmpeg_cmd = [
            "ffmpeg",
            "-ss",
            str(start_time),
            "-t",
            str(duration),
            "-i",
            video_file,
            "-vf",
            ",".join(video_filters),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-y",
            output_file,
        ]
        logger.debug(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")

        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
            logger.info(f"Clipped video with embedded subtitles saved to {output_file}")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed with exit code {e.returncode}: {e.stderr}")
            raise
        finally:
            os.unlink(tmp_srt_path)


def main():
    parser = argparse.ArgumentParser(
        description="Clip videos based on subtitle keywords."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the JSON config file"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug logging (overrides config)"
    )
    args = parser.parse_args()

    config = ClipperConfig.from_file(Path(args.config))
    logging_level = (
        getattr(logging, config.logging) if not args.verbose else logging.DEBUG
    )
    logger.setLevel(logging_level)

    input_dir = config.directory or "videos"
    output_dir = config.output_dir
    keywords = config.keywords
    word_alt_map = config.word_alt_map or {}
    pre_buffer = config.pre_buffer if config.pre_buffer is not None else 5.0
    post_buffer = config.post_buffer if config.post_buffer is not None else 5.0
    max_workers = config.max_workers if config.max_workers is not None else 1
    use_subdirs = config.use_subdirs

    video_files = [f for f in os.listdir(input_dir) if f.endswith(".mp4")]
    if not video_files:
        logger.warning(f"No .mp4 files found in {input_dir}")
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for video_file in video_files:
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
                )
            )
        for future in tqdm(futures, desc="Processing videos"):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error processing video: {e}")


if __name__ == "__main__":
    main()
