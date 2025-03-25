import logging
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple
import os
from pathlib import Path
import subprocess

import srt
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import spacy
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa

logger = logging.getLogger(__name__)

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")

# Initialize speech-to-text model with Wav2Vec2Processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
stt_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stt_model.to(device)

# Ensure NLTK WordNet data is downloaded
try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')

def sanitize_filename(filename: str, max_length: int = 200) -> str:
    """Sanitize a filename by removing invalid characters and emojis."""
    invalid_chars = r'[^a-zA-Z0-9_-]'
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    safe_name = re.sub(invalid_chars, "_", filename)
    safe_name = re.sub(r"\s+", "_", safe_name)
    safe_name = emoji_pattern.sub("", safe_name)
    safe_name = re.sub(r"_+", "_", safe_name)
    return safe_name[:max_length].strip("_").lower()

def load_subtitles_stream(subtitle_file: str) -> Iterable[srt.Subtitle]:
    """Load subtitles from an SRT file."""
    with open(subtitle_file, encoding="utf-8") as f:
        content = f.read()
    return srt.parse(content)

def preprocess_transcript(subtitle_file: str, video_file: str, tmp_dir: Path) -> List[srt.Subtitle]:
    subtitles = list(load_subtitles_stream(subtitle_file))
    censored_subs = [(i, sub) for i, sub in enumerate(subtitles) if r"[\h__\h]" in sub.content]
    logger.debug(rf"Found {len(censored_subs)} subtitles with '[\h__\h]' out of {len(subtitles)} total subtitles")
    
    if not censored_subs:
        logger.debug("No censored words found in transcript.")
        return subtitles

    # Extract audio from video
    video_base = sanitize_filename(os.path.basename(video_file))
    audio_path = tmp_dir / f"temp_audio_{video_base}.wav"
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', video_file,
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        str(audio_path),
        '-y'
    ]
    try:
        result = subprocess.run(
            ffmpeg_cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        logger.debug(f"Extracted audio to {audio_path} | FFmpeg output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg audio extraction failed: {e.stderr}")
        raise

    # Process each subtitle with censored words
    for idx, sub in censored_subs:
        start_time = sub.start.total_seconds()
        end_time = sub.end.total_seconds()
        duration = end_time - start_time
        content = sub.content
        logger.debug(f"Processing subtitle #{idx} at {start_time}-{end_time} (duration: {duration}s): '{content}'")

        censored_matches = list(re.finditer(r'\[\\h__\\h\]', content))
        logger.debug(rf"Detected {len(censored_matches)} '[\h__\h]' instances in '{content}'")
        if not censored_matches:
            continue

        num_censored = len(censored_matches)
        if num_censored == 1:
            segment_duration = max(0.5, duration)
            segment_path = tmp_dir / f"temp_segment_{idx}_0.wav"
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', str(audio_path),
                '-ss', str(start_time),
                '-t', str(segment_duration),
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                str(segment_path),
                '-y'
            ]
            try:
                result = subprocess.run(
                    ffmpeg_cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace'
                )
                logger.debug(f"Generated segment {segment_path} | FFmpeg output: {result.stdout}")
            except subprocess.CalledProcessError as e:
                logger.error(f"FFmpeg segment extraction failed: {e.stderr}")
                raise

            audio, sample_rate = librosa.load(segment_path, sr=16000)
            logger.debug(f"Loaded audio segment, length: {len(audio)} samples")
            if len(audio) == 0:
                logger.warning(f"Empty audio segment at {start_time}-{start_time + segment_duration}, skipping transcription")
                new_content = content
            else:
                input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values.to(device)
                with torch.no_grad():
                    logits = stt_model(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = processor.batch_decode(predicted_ids)[0].lower()
                logger.debug(f"Transcribed audio to: '{transcription}'")
                if transcription.strip():
                    new_content = content.replace(r"[\h__\h]", transcription, 1)
                    logger.debug(rf"Replaced [\h__\h] with '{transcription}', new content: '{new_content}'")
                else:
                    logger.warning(f"Transcription empty for segment {start_time}-{start_time + segment_duration}, keeping original content")
                    new_content = content
            subtitles[idx].content = new_content
            segment_path.unlink()
        else:
            words = re.split(r'\s+', content.strip())
            total_words = len(words)
            base_segment_duration = duration / max(total_words, num_censored) if duration > 0 else 0.5
            segment_duration = max(0.5, base_segment_duration)
            logger.debug(f"Multiple censored words detected, segment duration: {segment_duration}s")

            replacements = []
            for i, match in enumerate(censored_matches):
                word_idx = len(re.split(r'\s+', content[:match.start()].strip()))
                segment_start = start_time + (word_idx * segment_duration)
                segment_end = min(end_time, segment_start + segment_duration)

                if segment_end <= segment_start:
                    segment_end = segment_start + 0.5
                    if segment_end > end_time:
                        segment_end = end_time
                        segment_start = max(start_time, end_time - 0.5)

                segment_duration_adjusted = segment_end - segment_start
                segment_path = tmp_dir / f"temp_segment_{idx}_{i}.wav"
                ffmpeg_cmd = [
                    'ffmpeg',
                    '-i', str(audio_path),
                    '-ss', str(segment_start),
                    '-t', str(segment_duration_adjusted),
                    '-acodec', 'pcm_s16le',
                    '-ar', '16000',
                    '-ac', '1',
                    str(segment_path),
                    '-y'
                ]
                try:
                    result = subprocess.run(
                        ffmpeg_cmd,
                        check=True,
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        errors='replace'
                    )
                    logger.debug(f"Generated segment {segment_path} | FFmpeg output: {result.stdout}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"FFmpeg segment extraction failed: {e.stderr}")
                    raise

                audio, sample_rate = librosa.load(segment_path, sr=16000)
                logger.debug(f"Loaded audio segment, length: {len(audio)} samples")
                if len(audio) == 0:
                    logger.warning(f"Empty audio segment at {segment_start}-{segment_end}, using '[unknown]'")
                    transcription = "[unknown]"
                else:
                    input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values.to(device)
                    with torch.no_grad():
                        logits = stt_model(input_values).logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    transcription = processor.batch_decode(predicted_ids)[0].lower()
                    logger.debug(f"Transcribed audio to: '{transcription}'")
                replacements.append((match.start(), match.end(), transcription if transcription.strip() else "[unknown]"))
                segment_path.unlink()

            new_content = content
            for start, end, transcription in sorted(replacements, key=lambda x: x[0], reverse=True):
                new_content = new_content[:start] + transcription + new_content[end:]
                logger.debug(rf"Replaced [\h__\h] at position {start}-{end} with '{transcription}', new content: '{new_content}'")
            subtitles[idx].content = new_content

    audio_path.unlink()
    return subtitles

def find_keywords(subtitles: List[srt.Subtitle], search_targets: List[str]) -> List[srt.Subtitle]:
    """Find subtitle lines containing any of the search targets as whole words."""
    matched = []
    target_patterns = [re.compile(r'\b' + re.escape(target.lower()) + r'\b') for target in search_targets]
    for sub in subtitles:
        content_lower = sub.content.lower()
        for target, pattern in zip(search_targets, target_patterns):
            if pattern.search(content_lower):
                matched.append(sub)
                logger.debug(f"Matched subtitle: '{sub.content}' with target '{target}'")
                break
    return matched

def get_word_forms(word: str) -> Set[str]:
    """Generate alternative forms of a word using spaCy for POS and NLTK for lemmatization."""
    forms = {word.lower()}
    doc = nlp(word)
    if not doc:
        return forms
    token = doc[0]
    base = token.lemma_.lower()

    if token.pos_ == 'VERB':
        forms.update([base, base + 's', base + 'ing', base + 'ed'])
    elif token.pos_ == 'NOUN':
        if base.endswith(('s', 'x', 'z', 'ch', 'sh')):
            forms.add(base + 'es')
        else:
            forms.add(base + 's')
    elif token.pos_ == 'ADJ':
        forms.update([base, base + 'er', base + 'est'])

    for syn in wn.synsets(base):
        for lemma in syn.lemmas():
            forms.add(lemma.name().lower())
    return forms

def get_all_search_targets(keywords: Set[str], word_alt_map: Optional[Dict[str, List[str]]] = None) -> Tuple[List[str], Dict[str, Set[str]]]:
    """Generate search targets including keywords, alternatives, and NLP forms."""
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
    logger.debug(f"Search targets (including NLP forms): {all_targets}")
    logger.debug(f"Target to keywords mapping: {target_to_keywords}")
    return all_targets, target_to_keywords