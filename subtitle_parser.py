import srt
from pathlib import Path
from typing import List, Set, Optional, Dict, Tuple, Iterator
import logging
import inflect

logger = logging.getLogger(__name__)
p = inflect.engine()


def load_subtitles_stream(subtitle_file: Path) -> Iterator[srt.Subtitle]:
    """Stream subtitles from an SRT file."""
    try:
        with open(subtitle_file, "r", encoding="utf-8") as f:
            yield from srt.parse(f)
    except FileNotFoundError:
        logger.error(f"Subtitle file not found: {subtitle_file}")
        raise
    except srt.SRTParseError as e:
        logger.error(f"Invalid SRT format in {subtitle_file}: {e}")
        raise


def build_connected_components(word_alt_map: Dict[str, List[str]]) -> List[Set[str]]:
    """Build connected components from word_alt_map where edges are bidirectional."""
    parent = {word: word for word in word_alt_map}
    for word in word_alt_map:
        for alt in word_alt_map[word]:
            if alt not in parent:
                parent[alt] = alt

    def find(word: str) -> str:
        if parent[word] != word:
            parent[word] = find(parent[word])
        return parent[word]

    def union(word1: str, word2: str):
        root1 = find(word1)
        root2 = find(word2)
        if root1 != root2:
            parent[root1] = root2

    for word in word_alt_map:
        for alt in word_alt_map[word]:
            union(word, alt)

    components = {}
    for word in parent:
        root = find(word)
        if root not in components:
            components[root] = set()
        components[root].add(word)

    return list(components.values())


def generate_alternative_forms(word: str) -> Set[str]:
    """Generate alternative forms of a word using inflect."""
    forms = {word}

    plural = p.plural(word)  # type: ignore
    if plural and plural != word:
        forms.add(plural)

    singular = p.singular_noun(word)  # type: ignore
    if singular and singular != word:
        forms.add(singular)

    present_part = p.present_participle(word)  # type: ignore
    if present_part and present_part != word:
        forms.add(present_part)

    # past = p.past_tense(word)  # type: ignore
    # if past and past != word:
    #     forms.add(past)

    return forms


def get_all_search_targets(
    keywords: Set[str], word_alt_map: Optional[Dict[str, List[str]]]
) -> Tuple[List[str], Dict[str, Set[str]]]:
    """
    Generate all search targets from keywords and word_alt_map, along with a mapping to their originating keywords.

    Args:
        keywords: Set of keywords to search for.
        word_alt_map: Dictionary mapping words to their alternatives.

    Returns:
        Tuple containing:
        - List of search targets (keywords, alternatives, and their forms).
        - Dictionary mapping each search target to the set of keywords it relates to.
    """
    if not word_alt_map:
        word_alt_map = {}

    # Build connected components from the word_alt_map
    components = build_connected_components(word_alt_map) if word_alt_map else []

    # Identify components that contain at least one keyword and store their relevant keywords
    keyword_components = {}
    for component in components:
        relevant_keywords = component.intersection(keywords)
        if relevant_keywords:
            keyword_components[frozenset(component)] = relevant_keywords

    # Generate search targets and map them to their relevant keywords
    search_targets = set()
    target_to_keywords = {}
    for component, rel_keywords in keyword_components.items():
        for word in component:
            forms = generate_alternative_forms(word)
            search_targets.update(forms)
            for form in forms:
                if form not in target_to_keywords:
                    target_to_keywords[form] = set()
                target_to_keywords[form].update(rel_keywords)

    # Ensure all keywords are included as search targets, even if not in word_alt_map
    for keyword in keywords:
        if keyword not in search_targets:
            search_targets.add(keyword)
            target_to_keywords[keyword] = {keyword}
        else:
            target_to_keywords[keyword].add(keyword)

    return list(search_targets), target_to_keywords


def find_keywords(
    subtitles: List[srt.Subtitle], search_targets: List[str]
) -> List[srt.Subtitle]:
    """Find subtitles containing any of the search targets."""
    return [
        sub
        for sub in subtitles
        if any(target.lower() in sub.content.lower() for target in search_targets)
    ]
