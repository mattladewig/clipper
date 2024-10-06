# Clipper

Clipper is a Python script that processes video files by clipping segments based on keywords found in accompanying subtitle files. The script generates various outputs including clipped video, audio, subtitles, a frame image, and metadata.

## Features

- Clips video files based on keywords found in subtitle files.
- Configurable pre and post duration for clips.
- Generates a new folder with clipped video files.
- Outputs include:
  - Clipped video (`.mp4`)
  - Frame image (`.png`)
  - Audio (`.mp3`)
  - Subtitles (`.srt`)
  - Metadata (`.json`)

## Functional Requirements

1. The script should process all .mp4 files in the input folder.
2. Each .mp4 file should have a corresponding .srt file with the same name.
3. If the .srt file is missing, the script should log a warning and skip the video file.
4. The script should search for specified keywords in the .srt files.
5. If a keyword is found, the script should extract a clip from the video.
6. Search of keyword should use a fuzzy search matching and provide for a configurable match threshold.
7. The clip should start a specified number of seconds before the keyword's timestamp.
8. The clip should end a specified number of seconds after the keyword's timestamp.
9. The script should sanitize filenames to remove non-ASCII characters and replace spaces with underscores.
10. The script should extract a frame from the video at the keyword's timestamp.
11. The script should extract audio from the video for the duration of the clip.
12. The script should extract subtitles from the .srt file for the duration of the clip.
13. The script should save the extracted clip, frame, audio, and subtitles to the output folder.
14. The script should save metadata about the clip to a .json file.
15. The script should save the center point and keyword to a .txt file.
16. The script should log errors encountered during processing.
17. The script should accept command-line arguments for input folder, output folder, keywords, pre-duration, and post-duration.
18. The script should create the output folder if it does not exist.

## Dependencies

### Setting Up a Python Virtual Environment

It is recommended to use a Python virtual environment to manage dependencies and avoid conflicts with other projects. Follow these steps to set up and activate a virtual environment:

1. **Create a virtual environment**:

   ```sh
   python -m venv .venv
   ```

2. **Activate the virtual environment**:

   - On Windows:

     ```sh
     .\.venv\Scripts\activate
     ```

   - On macOS and Linux:

     ```sh
     source .venv/bin/activate
     ```

3. **Install the required packages**:

- Python 3.x
- `ffmpeg` (must be installed and available in the system PATH)
- `fuzzywuzzy` (for fuzzy keyword matching)
- `python-Levenshtein` (optional, for improved performance of `fuzzywuzzy`)

Install the required packages using pip and the provided `requirements.txt` file.

    ```sh
    pip install -r requirements.txt
    ```

1. **Run the script**:

   ```sh
   python clipper.py
   ```

## Usage

### Example Command

```sh
python clipper.py
```

### Example Script Usage

```python
# Example usage
input_folder = './mediaSource'
output_folder = './mediaClipOutput'
keywords = ['example', 'keyword']
pre_duration = 10  # seconds
post_duration = 10  # seconds

process_videos(input_folder, output_folder, keywords, pre_duration, post_duration)
```

### Use Cases

1. **Clipping Highlights from Videos**:
   - Extract highlights from sports events or lectures based on specific keywords.
2. **Creating Video Summaries**:

   - Generate summaries of long videos by clipping segments around important keywords.

3. **Content Creation**:
   - Create short clips for social media by extracting segments around trending keywords.

## Output Structure

The output folder will contain files named based on the original video file name and the keyword used to find the center point of the clip along with the centerpoint timestamp. For example:

```sh
mediaClipOutput/keyword1/
├── video1_example_00-01-23-456.mp4
├── video1_example_00-01-23-456.jpg
├── video1_example_00-01-23-456.mp3
├── video1_example_00-01-23-456.srt
└── video1_example_00-01-23-456.json
```

## Metadata

The `.json` metadata file includes information about the clip:

```json
{
  "center_point": "00:01:23,456",
  "keyword": "example",
  "start_time": "0:01:13",
  "duration": 20
}
```

## License

This project is licensed under the MIT License.

## Copyright

© 2024 Matt Ladewig
