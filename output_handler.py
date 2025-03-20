from pathlib import Path
import json
from typing import List, Dict, Any
import srt
import logging

logger = logging.getLogger(__name__)

def save_transcript(subtitles: List[srt.Subtitle], output_file: Path) -> None:
    """Save the transcript text to a file.

    Args:
        subtitles (List[srt.Subtitle]): List of subtitle objects.
        output_file (Path): Path to save the transcript.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for sub in subtitles:
            f.write(sub.content + '\n')
    logger.info(f"Transcript saved to {output_file}")

def save_metadata(metadata: Dict[str, Any], output_file: Path) -> None:
    """Save metadata to a JSON file.

    Args:
        metadata (Dict[str, Any]): Metadata to save.
        output_file (Path): Path to save the metadata.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Metadata saved to {output_file}")