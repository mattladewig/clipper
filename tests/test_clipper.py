import pytest
import os
import sys
import time
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from clipper import (
    generate_bidirectional_mapping,
    configure_logging,
    get_all_search_targets,
    sanitize_filename,
    timestamp_to_seconds,
    clip_video,
    clip_srt,
    validate_srt_content,
    add_srt_to_video,
    extract_frame,
    extract_audio,
    find_keyword_timestamps,
    get_video_duration,
    get_srt_duration,
    process_all_videos,
)


def test_generate_bidirectional_mapping():
    base_dict = {"a": ["b", "c"], "d": ["e"]}
    expected = {
        "a": ["b", "c"],
        "b": ["a", "c"],
        "c": ["a", "b"],
        "d": ["e"],
        "e": ["d"],
    }
    result = generate_bidirectional_mapping(base_dict)
    if result != expected:
        raise AssertionError(f"Expected {expected}, but got {result}")


def test_configure_logging(tmp_path):
    class Args:
        debug = True
        verbose = False

    args = Args()
    configure_logging(args)
    log_file = os.path.join("./log", "clipper.log")
    assert os.path.exists(log_file)


def test_get_all_search_targets():
    keywords = ["make", "tie", "study", "man"]
    expected = [
        "make",
        "tie",
        "study",
        "studies",
        "man",
        "making",
        "tying",
        "studied",
        "men",
    ]
    result = get_all_search_targets(keywords)
    assert set(result) == set(expected)


def test_sanitize_filename():
    if sanitize_filename("test file.txt") != "test_file.txt":
        raise AssertionError(
            f"Expected 'test_file.txt', but got {sanitize_filename("test file.txt")}"
        )
    if sanitize_filename("another:test/file.txt") != "another_test_file.txt":
        raise AssertionError(
            f"Expected 'another_test_file.txt', but got {sanitize_filename("another:test/file.txt")}"
        )


def test_timestamp_to_seconds():
    assert timestamp_to_seconds("00:01:30") == 90
    assert timestamp_to_seconds("01:00:00") == 3600
    assert timestamp_to_seconds("01:00:00,900") == 3600


def test_clip_video(mocker):
    mocker.patch("clipper.get_video_duration", return_value=3600)
    mocker.patch("clipper.ffmpeg.input")
    mocker.patch("clipper.ffmpeg.output")
    mocker.patch("clipper.ffmpeg.run")
    srt_content = """1
00:00:01,000 --> 00:00:02,000
Hello

2
00:00:03,000 --> 00:00:04,000
World
"""
    with open("./tests/test.srt", "w") as f:
        f.write(srt_content)
    clip_video(
        "./tests/test.mp4",
        "test transcript",
        "00:01:00",
        "00:02:00",
        5,
        10,
        "./output",
        "output.mp4",
    )


def test_clip_single_srt():
    srt_content = """1
00:00:01,000 --> 00:00:02,000
Hello

2
00:00:03,000 --> 00:00:04,000
World
"""
    with open("./tests/test.srt", "w") as f:
        f.write(srt_content)
    result = clip_srt("./tests/test.srt", 1, 2)
    expected = """1
00:00:01,000 --> 00:00:02,000
Hello
"""
    assert result.strip() == expected.strip()


def test_clip_multiple_srt():
    srt_content = """1
00:00:01,000 --> 00:00:02,000
Hello

2
00:00:03,000 --> 00:00:04,000
World
"""
    with open("./tests/test.srt", "w") as f:
        f.write(srt_content)
    result = clip_srt("./tests/test.srt", 1, 3)
    expected = """1
00:00:01,000 --> 00:00:02,000
Hello

2
00:00:03,000 --> 00:00:04,000
World
"""
    assert result.strip() == expected.strip()


def test_validate_srt_content():
    valid_srt = """1
00:00:01,000 --> 00:00:02,000
Hello

2
00:00:03,000 --> 00:00:04,000
World
"""
    invalid_srt = """1
00:00:01,000 -> 00:00:02,000
Hello

00:00:03,000 --> 00:00:04,000 World
"""
    assert validate_srt_content(valid_srt)
    assert not validate_srt_content(invalid_srt)


def test_add_srt_to_video(mocker):
    mocker.patch("clipper.ffmpeg.input")
    mocker.patch("clipper.ffmpeg.output")
    mocker.patch("clipper.ffmpeg.run")
    add_srt_to_video(
        "test.mp4", "1\n00:00:01,000 --> 00:00:02,000\nHello\n", "output", "output.mp4"
    )


def test_extract_frame(mocker):
    mocker.patch("clipper.ffmpeg.input")
    mocker.patch("clipper.ffmpeg.output")
    mocker.patch("clipper.ffmpeg.run")
    extract_frame("test.mp4", "00:01:00", "frame.png")


def test_extract_audio(mocker):
    mocker.patch("clipper.ffmpeg.input")
    mocker.patch("clipper.ffmpeg.output")
    mocker.patch("clipper.ffmpeg.run")
    extract_audio("test.mp4", 60, 30, "audio.mp3")


def test_find_keyword_timestamps():
    srt_content = """1
00:00:01,000 --> 00:00:02,000
Hello

2
00:00:03,000 --> 00:00:04,000
World
"""
    with open("./tests/test.srt", "w") as f:
        f.write(srt_content)
    result = find_keyword_timestamps("./tests/test.srt", ["Hello"], 90)
    expected = [
        ("./tests/test.srt", "00:00:01", "00:00:02", "Hello", "Hello", "Hello", 100)
    ]
    assert result == expected


def test_get_video_duration(mocker):
    mocker.patch("clipper.ffmpeg.probe", return_value={"format": {"duration": "3600"}})
    duration = get_video_duration("test.mp4")
    assert duration == 3600


def test_get_srt_duration():
    srt_content = """1
00:00:01,000 --> 00:00:02,000
Hello

2
00:00:03,000 --> 00:00:04,000
World
"""
    with open("./tests/test.srt", "w") as f:
        f.write(srt_content)
    duration = get_srt_duration("./tests/test.srt")
    assert duration == 3


def test_process_all_videos(mocker):
    mocker.patch("clipper.os.walk", return_value=[("./tests", [], ["test.mp4"])])
    mocker.patch(
        "clipper.find_keyword_timestamps",
        return_value=[
            ("./tests/test.srt", "00:00:01", "00:00:02", "Hello", "Hello", "Hello", 100)
        ],
    )
    mocker.patch("clipper.clip_video")
    result = process_all_videos(".", "./output", ["Hello"], 5, 10, 90, False)
    assert result == {"./tests/test.mp4": {"Hello": 1}}


exit_flag = threading.Event()


class ProcessMetaThread(threading.Thread):
    def run(self):
        while not exit_flag.is_set():
            time.sleep(0.1)


def test_process_meta_thread():
    exit_flag.clear()
    thread = ProcessMetaThread()
    thread.start()
    time.sleep(0.1)
    exit_flag.set()
    thread.join()
    assert not thread.is_alive()


if __name__ == "__main__":
    pytest.main()
