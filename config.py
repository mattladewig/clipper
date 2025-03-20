import json
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel


class ClipperConfig(BaseModel):
    video_file: str
    subtitle_file: str
    keywords: List[str]
    output_dir: str = "clips"
    verbose: bool = False
    directory: Optional[str] = None
    word_alt_map: Optional[Dict[str, List[str]]] = None
    pre_buffer: float = 20.0
    post_buffer: float = 20.0
    max_workers: Optional[int] = 4
    use_subdirs: bool = False  # New: Organize outputs into subdirectories

    @classmethod
    def from_file(cls, file_path: Path):
        with file_path.open("r") as f:
            data = json.load(f)
        return cls(**data)  # Correctly pass the data to the model
