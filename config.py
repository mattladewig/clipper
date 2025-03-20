import json
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, validator

class ClipperConfig(BaseModel):
    video_file: Optional[str] = None  # Changed to optional
    subtitle_file: Optional[str] = None  # Changed to optional
    keywords: List[str]
    output_dir: str = "clips"
    logging: str = "INFO"
    directory: Optional[str] = None
    word_alt_map: Optional[Dict[str, List[str]]] = None
    pre_buffer: Optional[float]
    post_buffer: Optional[float]
    max_workers: Optional[int]
    use_subdirs: bool = False

    @validator("logging")
    def validate_logging(cls, v):
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Logging level must be one of {valid_levels}, got {v}")
        return v.upper()

    @classmethod
    def from_file(cls, file_path: Path):
        with file_path.open("r") as f:
            data = json.load(f)
        return cls(**data)