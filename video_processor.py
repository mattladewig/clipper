import logging
# trunk-ignore(bandit/B404)
import subprocess
from pathlib import Path
from typing import List  # Fix import for List

import ffmpeg

logger = logging.getLogger(__name__)


def check_ffmpeg() -> None:
    """Check if FFmpeg is installed."""
    try:
        # trunk-ignore(bandit/B603)
        # trunk-ignore(bandit/B607)
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, check=True
        )
        logger.info(f"FFmpeg version: {result.stdout.splitlines()[0]}")
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg is not installed or not functioning: {e}")
        raise RuntimeError("FFmpeg is required to run this script.") from e
    except FileNotFoundError as e:
        logger.error("FFmpeg executable not found.")
        raise RuntimeError("FFmpeg is required to run this script.") from e


def clip_video(
    video_file: Path,
    start_time: float,
    end_time: float,
    output_file: Path,
    subtitle_file: Path,
    subtitle_content: List[str],
) -> None:
    """Clip a video segment and embed subtitle text."""
    try:
        duration = end_time - start_time
        # Create a temporary subtitle file for this clip
        temp_srt = output_file.with_suffix(".srt")
        with open(temp_srt, "w", encoding="utf-8") as f:
            for i, line in enumerate(subtitle_content, 1):
                end_offset = min(
                    end_time, start_time + 2.0
                )  # Assume 2-second display per line, adjust as needed
                f.write(f"{i}\n")
                f.write(f"{format_time(start_time)} --> {format_time(end_offset)}\n")
                f.write(f"{line}\n\n")
                start_time = end_offset  # Sequential display

        stream = ffmpeg.input(str(video_file), ss=start_time, t=duration).output(
            str(output_file),
            vf=f'subtitles="{str(temp_srt)}"',
            vcodec="libx264",
            acodec="aac",
            y=True,
        )
        stream.run(quiet=True)
        temp_srt.unlink()  # Clean up temporary SRT
        logger.info(f"Clipped video with embedded subtitles saved to {output_file}")
    except ffmpeg.Error as e:
        logger.error(f"Failed to clip video: {e}")
        raise


def clip_audio(
    video_file: Path, start_time: float, end_time: float, output_file: Path
) -> None:
    """Clip an audio segment from the video."""
    try:
        duration = end_time - start_time
        (
            ffmpeg.input(str(video_file), ss=start_time, t=duration)
            .output(str(output_file), c="copy")
            .run(overwrite_output=True, quiet=True)
        )
        logger.info(f"Clipped audio saved to {output_file}")
    except ffmpeg.Error as e:
        logger.error(f"Failed to clip audio: {e}")
        raise


def format_time(seconds: float) -> str:
    """Format time in SRT format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
