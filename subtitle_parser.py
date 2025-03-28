import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import librosa
import spacy
import srt
import torch
from fuzzywuzzy import process
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Configure logger explicitly to DEBUG level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False  # Disable propagation to parent loggers
if not logger.handlers:  # Only add handler if none exist
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Initialize NLP tools
nlp = spacy.load("en_core_web_sm")

censored_list = ["fuck", "fucking", "shit", "bitch", "nigger", "faggot", "moron", "beer", "bitchute", "cunt", "cock", "pussy", "tits", "motherfucker", "kike", "spic", "chink", "wop", "idiot", "retard", "cripple", "war", "kill", "tramp", "jap", "fuckwit", "fuckin' A"]

# Initialize speech-to-text model with Wav2Vec2Processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
stt_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stt_model.to(device)
logger.debug(f"Using device: {device}")


def sanitize_filename(filename: str, max_length: int = 200) -> str:
    """Sanitize a filename by removing invalid characters and emojis."""
    invalid_chars = r"[^a-zA-Z0-9_-]"
    emoji_pattern = re.compile(
        "[" "\U0001f600-\U0001f64f" "\U0001f300-\U0001f5ff" "\U0001f680-\U0001f6ff"
        "\U0001f1e0-\U0001f1ff" "\U00002702-\U000027b0" "\U000024c2-\U0001f251" "]+",
        flags=re.UNICODE,
    )
    safe_name = re.sub(invalid_chars, "_", filename)
    safe_name = re.sub(r"\s+", "_", safe_name)
    safe_name = emoji_pattern.sub("", safe_name)
    safe_name = re.sub(r"_+", "_", safe_name)
    return safe_name[:max_length].strip("_").lower()


def load_subtitles_stream(subtitle_file: str) -> Iterable[srt.Subtitle]:
    """Load subtitles from an SRT file."""
    logger.debug(f"Loading subtitles from {subtitle_file}")
    with open(subtitle_file, encoding="utf-8") as f:
        content = f.read()
    return srt.parse(content)


def preprocess_transcript(subtitle_file: str, video_file: str, tmp_dir: Path) -> List[srt.Subtitle]:
    """Process subtitles by replacing censored words with transcribed audio."""
    logger.debug(f"Starting preprocess_transcript for {video_file}")
    try:
        subtitles = list(load_subtitles_stream(subtitle_file))
    except Exception as e:
        logger.error(f"Failed to load subtitles from {subtitle_file}: {str(e)}")
        raise
    censored_subs = [(i, sub) for i, sub in enumerate(subtitles) if r"[\h__\h]" in sub.content]
    logger.debug(f"Found {len(censored_subs)} subtitles with '[\h__\h]' out of {len(subtitles)} total subtitles")

    if not censored_subs:
        logger.debug("No censored words found in transcript.")
        return subtitles

    audio_path = tmp_dir / f"temp_audio_{sanitize_filename(os.path.basename(video_file))}.wav"
    ffmpeg_cmd = ["ffmpeg", "-i", video_file, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(audio_path), "-y", "-loglevel", "error"]
    try:
        logger.debug(f"Running FFmpeg: {' '.join(ffmpeg_cmd)}")
        result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, encoding="utf-8")
        logger.debug(f"FFmpeg stdout: '{result.stdout}', stderr: '{result.stderr}'")
        if not audio_path.exists() or audio_path.stat().st_size < 1000:
            logger.error(f"Audio extraction failed: {audio_path} size: {audio_path.stat().st_size if audio_path.exists() else 0} bytes")
            raise RuntimeError("Audio extraction produced no usable output")
        logger.debug(f"Extracted audio to {audio_path}, size: {audio_path.stat().st_size} bytes")
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg audio extraction failed: stdout='{e.stdout}', stderr='{e.stderr}'")
        raise

    audio_duration = float(subprocess.run(
        ["ffprobe", "-i", str(audio_path), "-show_entries", "format=duration", "-v", "quiet", "-of", "csv=p=0"],
        check=True, capture_output=True, encoding="utf-8"
    ).stdout.strip())
    logger.debug(f"Audio duration: {audio_duration}s")

    for idx, sub in censored_subs:
        start_time = sub.start.total_seconds()
        end_time = sub.end.total_seconds()
        duration = end_time - start_time
        content = sub.content
        logger.debug(f"Processing subtitle #{idx} at {start_time}-{end_time} (duration: {duration}s): '{content}'")

        if start_time > audio_duration:
            logger.warning(f"Subtitle start time {start_time}s exceeds audio duration {audio_duration}s, skipping")
            continue

        censored_matches = list(re.finditer(r"\[\\h__\\h\]", content))
        logger.debug(f"Detected {len(censored_matches)} '[\h__\h]' instances in '{content}'")

        replacements = []
        for match_idx, match in enumerate(censored_matches):
            words = re.split(r"\s+", content.strip())
            censored_idx = len(re.split(r"\s+", content[:match.start()].strip()))
            total_words = len(words)
            logger.debug(f"Censored word at position {censored_idx} of {total_words} in '{content}'")

            # Widen segment window to ±0.5s around estimated position
            word_duration = duration / total_words
            word_start = start_time + (censored_idx * word_duration)
            segment_start = max(0, word_start - 0.5)
            segment_end = min(audio_duration, word_start + 0.5)
            segment_duration = segment_end - segment_start
            logger.debug(f"Estimated segment for censored word: {segment_start}-{segment_end} (duration: {segment_duration}s)")

            segment_path = tmp_dir / f"temp_segment_{idx}_{match_idx}.wav"
            debug_segment_path = tmp_dir / f"debug_segment_{idx}_{match_idx}.wav"
            ffmpeg_cmd = [
                "ffmpeg", "-i", str(audio_path), "-ss", str(segment_start), "-t", str(segment_duration),
                "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(segment_path), "-y", "-loglevel", "error"
            ]
            try:
                logger.debug(f"Running FFmpeg: {' '.join(ffmpeg_cmd)}")
                result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, encoding="utf-8")
                logger.debug(f"FFmpeg stdout: '{result.stdout}', stderr: '{result.stderr}'")
                if not segment_path.exists() or segment_path.stat().st_size <= 78:
                    logger.warning(f"Empty segment at {segment_start}-{segment_end}, size: {segment_path.stat().st_size if segment_path.exists() else 0} bytes")
                    replacements.append((match.start(), match.end(), "[unknown]"))
                else:
                    logger.debug(f"Generated segment {segment_path}, size: {segment_path.stat().st_size} bytes")
                    subprocess.run(["copy", str(segment_path), str(debug_segment_path)], shell=True)
                    logger.debug(f"Saved debug copy at {debug_segment_path}")

                    try:
                        audio, sr = librosa.load(segment_path, sr=16000)
                        logger.debug(f"Loaded audio segment, length: {len(audio)} samples, sample_rate: {sr}")
                        if len(audio) == 0:
                            logger.warning(f"Empty audio data from {segment_path}")
                            replacements.append((match.start(), match.end(), "[unknown]"))
                        else:
                            input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values.to(device)
                            logger.debug(f"Input values shape: {input_values.shape}")
                            with torch.no_grad():
                                logits = stt_model(input_values).logits
                                logger.debug(f"Logits shape: {logits.shape}")
                            transcription = processor.batch_decode(torch.argmax(logits, dim=-1))[0].lower()
                            logger.debug(f"Transcribed censored word audio to: '{transcription}'")

                            if not transcription.strip():
                                logger.debug("Empty transcription, relying on context")
                                best_candidate = "[unknown]"
                            else:
                                similarities = process.extract(transcription, censored_list)
                                best_word, audio_score = similarities[0]
                                logger.debug(f"Best transcription match: '{best_word}', audio similarity score: {audio_score}")
                                best_candidate = best_word if audio_score > 80 else "[unknown]"

                            candidates = {}
                            for word in censored_list:
                                candidate_content = content[:match.start()] + word + content[match.end():]
                                doc = nlp(candidate_content)
                                candidates[word] = 1.0 if doc.has_annotation("DEP") else 0.5
                            best_nlp_candidate = max(candidates, key=candidates.get)
                            logger.debug(f"Best NLP candidate: '{best_nlp_candidate}' with score: {candidates[best_nlp_candidate]}")

                            # Only use NLP if audio transcription fails completely
                            final_candidate = best_candidate if best_candidate != "[unknown]" else "[unknown]"
                            logger.debug(f"Final candidate selected: '{final_candidate}'")
                            replacements.append((match.start(), match.end(), final_candidate))
                    except Exception as e:
                        logger.error(f"Transcription failed for {segment_path}: {str(e)}")
                        replacements.append((match.start(), match.end(), "[stt_error]"))
            except subprocess.CalledProcessError as e:
                logger.error(f"FFmpeg segment extraction failed: stdout='{e.stdout}', stderr='{e.stderr}'")
                replacements.append((match.start(), match.end(), "[error]"))
            finally:
                if segment_path.exists():
                    logger.debug(f"Cleaning up {segment_path}")
                    segment_path.unlink()

        new_content = content
        for start, end, replacement in sorted(replacements, key=lambda x: x[0], reverse=True):
            new_content = new_content[:start] + replacement + new_content[end:]
            logger.debug(f"Replaced '[\h__\h]' with '{replacement}' at {start}-{end}, new content: '{new_content}'")
        subtitles[idx].content = new_content

    if audio_path.exists():
        logger.debug(f"Cleaning up {audio_path}")
        audio_path.unlink()
    logger.debug("Finished preprocess_transcript")
    return subtitles

def find_keywords(subtitles: List[srt.Subtitle], search_targets: List[str]) -> List[srt.Subtitle]:
    """Find subtitle lines containing any of the search targets as whole words."""
    logger.debug(f"Finding keywords in subtitles with targets: {search_targets}")
    matched = []
    target_patterns = [re.compile(r"\b" + re.escape(target.lower()) + r"\b") for target in search_targets]
    for sub in subtitles:
        content_lower = sub.content.lower()
        for target, pattern in zip(search_targets, target_patterns):
            if pattern.search(content_lower):
                matched.append(sub)
                logger.debug(f"Matched subtitle: '{sub.content}' with target '{target}'")
                break
    return matched


def get_word_forms(word: str) -> Set[str]:
    """Generate alternative forms of a word using spaCy for POS and lemmatization, without synonyms."""
    logger.debug(f"Generating word forms for: '{word}'")
    forms = {word.lower()}
    doc = nlp(word)
    if not doc:
        logger.debug(f"No SpaCy tokens for '{word}'")
        return forms
    token = doc[0]
    base = token.lemma_.lower()
    forms.add(base)

    if token.pos_ == "VERB":
        forms.update([base + "s", base + "ing", base + "ed"])
        if base == "be":
            forms.update(["is", "are", "was", "were", "being", "been"])
        elif base == "have":
            forms.update(["has", "having", "had"])
        elif base == "do":
            forms.update(["does", "doing", "did"])
    elif token.pos_ == "NOUN":
        if base.endswith(("s", "x", "z", "ch", "sh")):
            forms.add(base + "es")
        elif base.endswith("y") and base[-2] not in "aeiou":
            forms.add(base[:-1] + "ies")
        else:
            forms.add(base + "s")
    elif token.pos_ == "ADJ":
        forms.update([base + "er", base + "est"])
        if base == "good":
            forms.update(["better", "best"])
        elif base == "bad":
            forms.update(["worse", "worst"])

    logger.debug(f"Generated forms: {forms}")
    return forms


def get_all_search_targets(keywords: Set[str], word_alt_map: Optional[Dict[str, List[str]]] = None) -> Tuple[List[str], Dict[str, Set[str]]]:
    """Generate search targets including keywords, explicit alternatives, and morphological forms."""
    logger.debug(f"Generating search targets for keywords: {keywords}")
    word_alt_map = word_alt_map or {}
    search_targets = set(keywords)
    target_to_keywords = {kw: {kw} for kw in keywords}

    for keyword in keywords:
        if keyword in word_alt_map:
            alts = set(word_alt_map[keyword])
            search_targets.update(alts)
            target_to_keywords[keyword].update(alts)
            for alt in alts:
                target_to_keywords.setdefault(alt, set()).add(keyword)

    for keyword in keywords:
        forms = get_word_forms(keyword)
        search_targets.update(forms)
        target_to_keywords[keyword].update(forms)
        for form in forms:
            target_to_keywords.setdefault(form, set()).add(keyword)

    all_targets = sorted(list(search_targets))
    logger.debug(f"Search targets (including morphological forms): {all_targets}")
    logger.debug(f"Target to keywords mapping: {target_to_keywords}")
    return all_targets, target_to_keywords