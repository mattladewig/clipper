from pydantic import BaseModel
from typing import List, Optional, Dict

class ClipperConfig(BaseModel):
    video_file: str
    subtitle_file: str
    keywords: List[str]
    output_dir: str = "output"
    verbose: bool = False
    directory: Optional[str] = None
    word_alt_map: Optional[Dict[str, List[str]]] = None
    pre_buffer: float = 20.0
    post_buffer: float = 20.0
    max_workers: Optional[int] = 4
    use_subdirs: bool = False  # New: Organize outputs into subdirectories