from pathlib import Path
import logging
import subprocess
from typing import List

logger = logging.getLogger(__name__)


def check_ffmpeg() -> None:
    """Check if FFmpeg is installed."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg is not installed or not functioning: {e}")
        raise RuntimeError("FFmpeg is required to run this script.") from e


def parse_srt_time(time_str: str) -> float:
    """Convert SRT time (HH:MM:SS,mmm) to seconds."""
    hours, minutes, seconds = map(float, time_str.replace(",", ".").split(":"))
    return hours * 3600 + minutes * 60 + seconds


def format_time(seconds: float) -> str:
    """Format time in SRT format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def clip_video(
    video_file: Path,
    start_time: float,
    end_time: float,
    output_file: Path,
    subtitle_file: Path,
    subtitle_content: List[str],
) -> None:
    """Clip a video segment and embed subtitle text for the given time range."""
    try:
        duration = end_time - start_time
        if duration <= 0:
            logger.error(
                f"Invalid duration ({duration}) for {video_file}: start_time={start_time}, end_time={end_time}"
            )
            raise ValueError("Invalid clip duration")

        temp_srt = output_file.with_suffix(".srt")
        with open(temp_srt, "w", encoding="utf-8") as f:
            for i, entry in enumerate(subtitle_content, 1):
                # Expecting full SRT entry format: "index\nstart --> end\ntext"
                lines = entry.strip().split("\n")
                if len(lines) < 3:
                    logger.debug(f"Skipping malformed SRT entry: {entry}")
                    continue
                try:
                    start_str, end_str = lines[1].split(" --> ")
                    sub_start = parse_srt_time(start_str)
                    sub_end = parse_srt_time(end_str)
                    text = "\n".join(lines[2:])
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse SRT entry: {entry} - {e}")
                    continue
                # Filter subtitles within the clip’s time range
                if sub_end <= start_time or sub_start >= end_time:
                    continue
                # Adjust timings relative to clip start
                rel_start = max(0, sub_start - start_time)
                rel_end = min(duration, sub_end - start_time)
                if rel_start >= rel_end:
                    continue  # Skip if subtitle duration becomes invalid
                f.write(f"{i}\n")
                f.write(f"{format_time(rel_start)} --> {format_time(rel_end)}\n")
                f.write(f"{text}\n\n")
        srt_content = temp_srt.read_text()
        logger.debug(f"Temporary SRT for {output_file}:\n{srt_content}")
        if not srt_content.strip():
            logger.warning(
                f"No subtitles overlap with clip range {start_time}-{end_time} for {output_file}"
            )
            # Proceed without subtitles if none match

        output_file_str = str(output_file)
        if not output_file_str:
            logger.error(f"Output file path is empty for {video_file}")
            temp_srt.unlink()
            raise ValueError("Output file path cannot be empty")
        logger.debug(f"Output file path: {output_file_str}")

        cmd = [
            "ffmpeg",
            "-ss",
            str(start_time),
            "-t",
            str(duration),
            "-i",
            str(video_file),
            "-vf",
            f"subtitles={str(temp_srt)}" if srt_content.strip() else "",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-y",
            output_file_str,
        ]
        if not srt_content.strip():
            cmd.remove("-vf")  # Remove empty -vf if no subtitles
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")

        process = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
        )
        if process.returncode != 0:
            error_msg = process.stderr or "No error output"
            logger.error(
                f"FFmpeg failed with exit code {process.returncode}: {error_msg}"
            )
            raise RuntimeError(f"FFmpeg failed: {error_msg}")

        temp_srt.unlink()
        logger.info(f"Clipped video with embedded subtitles saved to {output_file}")
    except subprocess.TimeoutExpired:
        logger.error(f"FFmpeg timed out after 60s for {video_file}")
        temp_srt.unlink()
        raise
    except Exception as e:
        logger.error(f"Unexpected error in clip_video: {e}")
        temp_srt.unlink()
        raise


def clip_audio(
    video_file: Path, start_time: float, end_time: float, output_file: Path
) -> None:
    """Clip an audio segment from the video."""
    try:
        duration = end_time - start_time
        output_file_str = str(output_file)
        if not output_file_str:
            logger.error(f"Output file path is empty for {video_file} in clip_audio")
            raise ValueError("Output file path cannot be empty")
        logger.debug(f"Audio output file path: {output_file_str}")

        cmd = [
            "ffmpeg",
            "-ss",
            str(start_time),
            "-t",
            str(duration),
            "-i",
            str(video_file),
            "-c",
            "copy",
            "-y",
            output_file_str,
        ]
        logger.debug(f"FFmpeg audio command: {' '.join(cmd)}")

        process = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
        )
        if process.returncode != 0:
            error_msg = process.stderr or "No error output"
            logger.error(
                f"FFmpeg audio failed with exit code {process.returncode}: {error_msg}"
            )
            raise RuntimeError(f"FFmpeg audio failed: {error_msg}")

        logger.info(f"Clipped audio saved to {output_file}")
    except subprocess.TimeoutExpired:
        logger.error(f"FFmpeg timed out after 60s for {video_file} in clip_audio")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in clip_audio: {e}")
        raise
