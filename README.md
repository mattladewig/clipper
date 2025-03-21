# Clipper

Clipper is a Python-based tool designed to extract video clips from `.mp4` files based on keywords found in corresponding `.srt` subtitle files. It processes videos, identifies subtitle ranges containing specified keywords, and generates clips with embedded subtitles at a fixed 404x720 resolution. Output filenames reflect all matched keywords within each clip’s subtitle range, making it easy to identify content.

## Features
- **Keyword-Based Clipping**: Extracts video segments where specified keywords appear in subtitles.
- **Multi-Keyword Filenames**: Output filenames include all unique keywords found in a clip’s subtitle range (e.g., `2-big_copper_001_111.mp4`).
- **Subtitle Embedding**: Embeds adjusted subtitles into each clip using FFmpeg.
- **Configurable Buffers**: Adds pre- and post-buffers (default 5 seconds) around matched subtitle timings.
- **Resolution Control**: Scales clips to 404x720 while preserving aspect ratio.
- **Parallel Processing**: Supports multi-threaded video processing via a configurable thread pool.
- **Flexible Configuration**: Uses a JSON config file for keywords, directories, and settings.

## Requirements
- Python 3.6+
- FFmpeg (installed and accessible in your PATH)
- Required Python packages:
  - `srt`
  - `tqdm`
  - Install via: `pip install -r requirements.txt`

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/mattladewig/clipper.git
   cd clipper
   ```
2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Install FFmpeg**:
   - On Ubuntu: `sudo apt install ffmpeg`
   - On macOS: `brew install ffmpeg`
   - On Windows: Download from [FFmpeg’s site](https://ffmpeg.org/download.html) and add to PATH.

## Usage
1. **Prepare Your Files**:
   - Place `.mp4` videos and their corresponding `.srt` subtitle files (named identically, e.g., `1.mp4` and `1.srt`) in the `videos/` directory (or configure a custom directory in `config.json`).

2. **Configure the Tool**:
   - Edit `config.json` to specify keywords and settings (see [Configuration](#configuration) below).
   - Example `config.json`:
     ```json
     {
       "directory": "videos",
       "output_dir": "clips",
       "keywords": ["big", "copper"],
       "word_alt_map": {},
       "pre_buffer": 5.0,
       "post_buffer": 5.0,
       "max_workers": 2,
       "use_subdirs": false,
       "logging": "INFO"
     }
     ```

3. **Run the Script**:
   ```bash
   python clipper.py --config config.json
   ```
   - Add `--verbose` for debug-level logging: `python clipper.py --config config.json --verbose`.

4. **Output**:
   - Clips are saved in the `clips/` directory (or as configured), named like `videoID-keywords_clipNumber_startTime.mp4` (e.g., `2-big_copper_001_111.mp4`).

## Configuration
The `config.json` file supports the following options:
- **`directory`**: Input directory for `.mp4` and `.srt` files (default: `"videos"`).
- **`output_dir`**: Output directory for clipped videos (default: `"clips"`).
- **`keywords`**: List of keywords to search for in subtitles (e.g., `["big", "copper"]`).
- **`word_alt_map`**: Dictionary of keyword aliases (e.g., `{"big": ["large", "huge"]}`). Optional.
- **`pre_buffer`**: Seconds added before each matched subtitle (default: `5.0`).
- **`post_buffer`**: Seconds added after each matched subtitle (default: `5.0`).
- **`max_workers`**: Number of threads for parallel processing (default: `1`).
- **`use_subdirs`**: If `true`, creates subdirectories per video ID in the output directory (default: `false`).
- **`logging`**: Logging level (`"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, default: `"INFO"`).

## Example Output
Given:
- `videos/1.mp4` and `videos/1.srt`
- `videos/2.mp4` and `videos/2.srt`
- Keywords: `["big", "copper"]`

Running `python clipper.py --config config.json` might produce:
```
clips/1-big_001_11.mp4
clips/1-big_002_46.mp4
clips/1-big_003_156.mp4
clips/1-big_004_185.mp4
clips/1-copper_005_333.mp4
clips/2-big_copper_001_111.mp4
```

Each clip:
- Is scaled to 720p.
- Contains embedded subtitles adjusted to the clip’s timeline.
- Has a filename reflecting all keywords in the clip’s subtitle range.

## How It Works
1. **Subtitle Parsing**: Loads `.srt` files and searches for keywords (case-insensitive).
2. **Range Merging**: Combines overlapping subtitle ranges with buffers.
3. **Clip Extraction**: Uses FFmpeg to cut video segments and embed subtitles.
4. **Naming**: Generates filenames based on all keywords found in the clip’s full subtitle range (`clip_subs_filtered`).

## Debugging
- Use `--verbose` to see detailed logs:
  ```bash
  python clipper.py --config config.json --verbose
  ```
- Logs include search targets, clip ranges, subtitle contents, and FFmpeg commands.

## Contributing
Feel free to fork the repository, submit issues, or create pull requests on [GitHub](https://github.com/mattladewig/clipper).

## License
This project is open-source under the [MIT License](LICENSE).

## Acknowledgments
- Built with Python, FFmpeg, and the `srt` library.
- Thanks to contributors and users for feedback!

