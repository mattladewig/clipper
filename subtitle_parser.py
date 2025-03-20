import srt
from typing import List, Set, Dict, Iterable
import logging

logger = logging.getLogger(__name__)

def load_subtitles_stream(subtitle_file: str) -> Iterable[srt.Subtitle]:
    """Load subtitles from an SRT file."""
    with open(subtitle_file, encoding="utf-8") as f:
        return srt.parse(f)

def find_keywords(subtitles: List[srt.Subtitle], search_targets: List[str]) -> List[srt.Subtitle]:
    """Find subtitle lines containing any of the search targets."""
    matched = []
    search_targets_lower = [target.lower() for target in search_targets]
    for sub in subtitles:
        content_lower = sub.content.lower()
        for target in search_targets_lower:
            if target in content_lower:
                matched.append(sub)
                logger.debug(f"Matched subtitle: '{sub.content}' with target '{target}'")
                break  # Stop checking other targets once matched
    return matched

def get_all_search_targets(keywords: Set[str], word_alt_map: Dict[str, List[str]] = None) -> tuple[List[str], Dict[str, Set[str]]]:
    """
    Generate a list of all search targets including keywords and their alternatives,
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

    all_targets = list(search_targets)
    logger.debug(f"Search targets: {all_targets}")
    logger.debug(f"Target to keywords mapping: {target_to_keywords}")
    return all_targets, target_to_keywords