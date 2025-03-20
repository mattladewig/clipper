import argparse
from pathlib import Path
import logging
from typing import Set, Optional, Dict, List, Tuple
from config import ClipperConfig
from subtitle_parser import load_subtitles_stream, find_keywords, get_all_search_targets
from video_processor import check_ffmpeg, clip_video, clip_audio
from output_handler import save_transcript, save_metadata
from tqdm import tqdm
import srt
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import ffmpeg

logger = logging.getLogger(__name__)

def setup_logging(verbose: bool) -> None:
    """Set up logging with the specified verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

def get_video_duration(video_file: Path) -> float:
    """Get video duration in seconds using FFmpeg probe."""
    try:
        probe = ffmpeg.probe(str(video_file))
        return float(probe['format']['duration'])
    except ffmpeg.Error as e:
        logger.error(f"Failed to probe duration for {video_file}: {e}")
        return 10800.0  # Default to 3 hours if probing fails

def merge_subtitle_ranges(matched_subtitles: List[srt.Subtitle], pre_buffer: float, post_buffer: float) -> List[Tuple[float, float, List[srt.Subtitle]]]:
    """Merge overlapping subtitle time ranges with buffers."""
    if not matched_subtitles:
        return []

    events = sorted(
        [(sub.start.total_seconds() - pre_buffer, sub.end.total_seconds() + post_buffer, sub)
         for sub in matched_subtitles],
        key=lambda x: x[0]
    )

    merged = []
    current_start, current_end, current_subs = events[0][0], events[0][1], [events[0][2]]

    for start, end, sub in events[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            current_subs.append(sub)
        else:
            merged.append((current_start, current_end, current_subs))
            current_start, current_end, current_subs = start, end, [sub]

    merged.append((current_start, current_end, current_subs))
    return merged

def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """Sanitize and truncate filename to meet Windows/Linux requirements."""
    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    filename = "".join(c if c in safe_chars else "_" for c in filename)
    max_base_length = max_length - 4  # Account for extension
    return filename[:max_base_length] if len(filename) > max_base_length else filename

def process_video(video_file: Path, subtitle_file: Path, keywords: Set[str], search_targets: List[str], target_to_keywords: Dict[str, Set[str]], output_dir: Path,
                  pre_buffer: float = 20.0, post_buffer: float = 20.0, use_subdirs: bool = False) -> None:
    """
    Process a single video file, creating clips based on subtitles containing search targets.

    Args:
        video_file: Path to the video file.
        subtitle_file: Path to the corresponding SRT file.
        keywords: Set of original keywords.
        search_targets: List of expanded search targets.
        target_to_keywords: Mapping from search targets to their associated keywords.
        output_dir: Directory to save output files.
        pre_buffer: Seconds to include before subtitle start.
        post_buffer: Seconds to include after subtitle end.
        use_subdirs: Whether to organize outputs into subdirectories.
    """
    video_duration = get_video_duration(video_file)
    base_output_dir = output_dir
    if use_subdirs:
        output_dir = base_output_dir / video_file.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    subtitles = list(load_subtitles_stream(subtitle_file))
    matched_subtitles = find_keywords(subtitles, search_targets)

    if not matched_subtitles:
        logger.warning(f"No search targets found in {subtitle_file}")
        return

    merged_clips = merge_subtitle_ranges(matched_subtitles, pre_buffer, post_buffer)
    video_name = video_file.stem

    for i, (start_time, end_time, clip_subs) in enumerate(tqdm(merged_clips, desc=f"Processing {video_file.name}")):
        start_time = max(0, start_time)
        end_time = min(end_time, video_duration)

        # Identify which search targets are present in the clip's subtitles
        present_targets = {target for sub in clip_subs for target in search_targets if target.lower() in sub.content.lower()}
        matched_keywords = set()
        for target in present_targets:
            matched_keywords.update(target_to_keywords.get(target, set()))
        keyword_prefix = "-".join(sorted(matched_keywords)) if matched_keywords else "unknown"

        base_name = f"{video_name}-{keyword_prefix}_{i:03d}_{int(start_time)}"
        safe_base_name = sanitize_filename(base_name, max_length=255)

        output_file = output_dir / f"{safe_base_name}.mp4"
        audio_file = output_dir / f"{safe_base_name}.mp3"
        transcript_file = output_dir / f"{safe_base_name}.txt"
        metadata_file = output_dir / f"{safe_base_name}.json"

        clip_video(video_file, start_time, end_time, output_file, subtitle_file, [sub.content for sub in clip_subs])
        clip_audio(video_file, start_time, end_time, audio_file)
        save_transcript(clip_subs, transcript_file)
        metadata = {
            "source_file": str(video_file),
            "keywords": list(matched_keywords),
            "start_time": start_time,
            "end_time": end_time,
            "transcript": [sub.content for sub in clip_subs]
        }
        save_metadata(metadata, metadata_file)
        logger.info(f"Created clip {output_file.name} with keywords {list(matched_keywords)} from {video_file} at {start_time}-{end_time}")

def process_video_wrapper(args: Tuple[Path, Path, Set[str], List[str], Dict[str, Set[str]], Path, float, float, bool], max_retries: int = 3) -> None:
    """Wrapper for parallel processing with retries."""
    video_file, subtitle_file, keywords, search_targets, target_to_keywords, output_dir, pre_buffer, post_buffer, use_subdirs = args
    for attempt in range(max_retries):
        try:
            process_video(video_file, subtitle_file, keywords, search_targets, target_to_keywords, output_dir, pre_buffer, post_buffer, use_subdirs)
            return
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for {video_file}: {e}")
            if attempt == max_retries - 1:
                logger.error(f"Max retries reached for {video_file}: {e}")

def load_checkpoint(checkpoint_file: Path) -> Set[str]:
    """Load completed files from checkpoint."""
    if checkpoint_file.exists():
        with checkpoint_file.open('r') as f:
            return set(line.strip() for line in f)
    return set()

def save_checkpoint(checkpoint_file: Path, video_file: str) -> None:
    """Append completed file to checkpoint."""
    with checkpoint_file.open('a') as f:
        f.write(f"{video_file}\n")

def get_adaptive_workers(max_workers: Optional[int]) -> int:
    """Determine optimal number of workers based on system resources."""
    cpu_count = multiprocessing.cpu_count()
    return min(max_workers or cpu_count, cpu_count)

def main():
    parser = argparse.ArgumentParser(description="Clipper: Extract video segments based on subtitles.")
    parser.add_argument("video_file", type=Path, nargs="?", help="Path to the video file")
    parser.add_argument("subtitle_file", type=Path, nargs="?", help="Path to the SRT subtitle file")
    parser.add_argument("keywords", nargs='*', help="Keywords to search for")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--directory", type=Path, help="Source directory with videos for batch processing")
    parser.add_argument("--config", type=Path, help="Path to configuration file")
    parser.add_argument("--pre-buffer", type=float, default=20.0, help="Seconds before subtitle start")
    parser.add_argument("--post-buffer", type=float, default=20.0, help="Seconds after subtitle end")
    parser.add_argument("--max-workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--use-subdirs", action="store_true", help="Organize outputs into subdirectories")

    args = parser.parse_args()

    word_alt_map = None
    pre_buffer = args.pre_buffer
    post_buffer = args.post_buffer
    max_workers = args.max_workers
    use_subdirs = args.use_subdirs

    if args.config:
        config = ClipperConfig.parse_file(args.config)
        args.video_file = Path(config.video_file) if config.video_file else args.video_file
        args.subtitle_file = Path(config.subtitle_file) if config.subtitle_file else args.subtitle_file
        args.keywords = config.keywords if config.keywords else args.keywords
        args.output_dir = Path(config.output_dir) if config.output_dir else args.output_dir
        args.verbose = config.verbose if config.verbose is not None else args.verbose
        args.directory = Path(config.directory) if config.directory else args.directory
        word_alt_map = config.word_alt_map
        pre_buffer = config.pre_buffer if config.pre_buffer is not None else args.pre_buffer
        post_buffer = config.post_buffer if config.post_buffer is not None else args.post_buffer
        max_workers = config.max_workers if config.max_workers is not None else args.max_workers
        use_subdirs = config.use_subdirs if config.use_subdirs is not None else args.use_subdirs

    setup_logging(args.verbose)
    check_ffmpeg()

    keywords_set = set(args.keywords)
    # Generate search targets and mapping once for all videos
    search_targets, target_to_keywords = get_all_search_targets(keywords_set, word_alt_map)
    max_workers = get_adaptive_workers(max_workers)

    if args.directory:
        if not args.directory.exists() or not args.directory.is_dir():
            logger.error(f"Directory {args.directory} does not exist or is not a directory")
            return
        logger.info(f"Processing all videos in directory: {args.directory} with {max_workers} workers")

        checkpoint_file = args.output_dir / "checkpoint.txt"
        completed_files = load_checkpoint(checkpoint_file)

        video_files = list(args.directory.glob("*.mp4"))
        tasks = [
            (video_file, video_file.with_suffix(".srt"), keywords_set, search_targets, target_to_keywords, args.output_dir, pre_buffer, post_buffer, use_subdirs)
            for video_file in video_files if video_file.with_suffix(".srt").exists() and str(video_file) not in completed_files
        ]

        progress_log = args.output_dir / "progress.log"
        with open(progress_log, 'a') as log:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {executor.submit(process_video_wrapper, task): task[0] for task in tasks}
                for future in tqdm(as_completed(future_to_file), total=len(tasks), desc="Total Progress", file=log):
                    video_file = future_to_file[future]
                    try:
                        future.result()
                        save_checkpoint(checkpoint_file, str(video_file))
                        log.write(f"Completed {video_file}\n")
                    except Exception as e:
                        logger.error(f"Failed to process {video_file}: {e}")
    else:
        if not args.video_file or not args.subtitle_file or not args.video_file.exists() or not args.subtitle_file.exists():
            logger.error("Video file or subtitle file does not exist or was not provided")
            return
        process_video(args.video_file, args.subtitle_file, keywords_set, search_targets, target_to_keywords, args.output_dir, pre_buffer, post_buffer, use_subdirs)

if __name__ == "__main__":
    main()