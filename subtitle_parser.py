import srt
from typing import List, Set, Dict, Iterable, Optional
import logging
import re

logger = logging.getLogger(__name__)

def load_subtitles_stream(subtitle_file: str) -> Iterable[srt.Subtitle]:
    """Load subtitles from an SRT file."""
    with open(subtitle_file, encoding="utf-8") as f:
        content = f.read()
    return srt.parse(content)

def find_keywords(subtitles: List[srt.Subtitle], search_targets: List[str]) -> List[srt.Subtitle]:
    """Find subtitle lines containing any of the search targets as whole words."""
    matched = []
    # Compile regex patterns for whole-word matching
    target_patterns = [re.compile(r'\b' + re.escape(target.lower()) + r'\b') for target in search_targets]
    for sub in subtitles:
        content_lower = sub.content.lower()
        for target, pattern in zip(search_targets, target_patterns):
            if pattern.search(content_lower):
                matched.append(sub)
                logger.debug(f"Matched subtitle: '{sub.content}' with target '{target}'")
                break  # Stop checking other targets once matched
    return matched

def get_all_search_targets(keywords: Set[str], word_alt_map: Optional[Dict[str, List[str]]] = None) -> tuple[List[str], Dict[str, Set[str]]]:
    """
    Generate a list of all unique search targets including keywords and their alternatives,
    and a mapping from targets to original keywords.
    """
    word_alt_map = word_alt_map or {}
    search_targets = set(keywords)
    target_to_keywords = {kw: {kw} for kw in keywords}

    for keyword in keywords:
        if keyword in word_alt_map:
            alts = set(word_alt_map[keyword])
            search_targets.update(alts)
            target_to_keywords[keyword].update(alts)
            for alt in alts:
                if alt not in target_to_keywords:
                    target_to_keywords[alt] = {keyword}
                else:
                    target_to_keywords[alt].add(keyword)

    all_targets = sorted(list(search_targets))  # Flatten and deduplicate explicitly
    logger.debug(f"Search targets (flattened and unique): {all_targets}")
    logger.debug(f"Target to keywords mapping: {target_to_keywords}")
    return all_targets, target_to_keywords