"""
DarkID Stylometric Feature Extraction Module
============================================
Extracts the canonical 6 function-word stylometric frequencies per 100 words
using lexicon-based regular expressions from English_Function_Words_Set.

Features:
  1. AuxiliaryVerbs_freq_per_100_words
  2. Conjunctions_freq_per_100_words
  3. Determiners_freq_per_100_words
  4. Prepositions_freq_per_100_words
  5. Pronouns_freq_per_100_words
  6. Quantifiers_freq_per_100_words
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Locate lexicon directory relative to project root
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
LEXICON_DIR = PROJECT_DIR / "English_Function_Words_Set"

FEATURE_NAMES = [
    "AuxiliaryVerbs_freq_per_100_words",
    "Conjunctions_freq_per_100_words",
    "Determiners_freq_per_100_words",
    "Prepositions_freq_per_100_words",
    "Pronouns_freq_per_100_words",
    "Quantifiers_freq_per_100_words",
]

FEATURE_DISPLAY_NAMES = {
    "AuxiliaryVerbs_freq_per_100_words": "Auxiliary Verbs",
    "Conjunctions_freq_per_100_words": "Conjunctions",
    "Determiners_freq_per_100_words": "Determiners",
    "Prepositions_freq_per_100_words": "Prepositions",
    "Pronouns_freq_per_100_words": "Pronouns",
    "Quantifiers_freq_per_100_words": "Quantifiers",
}

TOKEN_PATTERN = re.compile(r"\b[\w']+\b")

# Cache compiled patterns once in memory
_PATTERNS: Dict[str, re.Pattern] = {}
_CATEGORIES: Dict[str, set] = {}


def _init_lexicons() -> None:
    """Load and compile function-word regex patterns from lexicons."""
    global _PATTERNS, _CATEGORIES

    if _PATTERNS:
        return

    if not LEXICON_DIR.exists():
        raise FileNotFoundError(
            f"Lexicon directory not found at: {LEXICON_DIR}. "
            "Ensure English_Function_Words_Set is present in the project root."
        )

    for txt_file in sorted(LEXICON_DIR.glob("*.txt")):
        category = txt_file.stem.replace("English", "")
        words = set()

        with open(txt_file, "r", encoding="cp1252", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                words.add(line.lower())

        _CATEGORIES[category] = words

        # Longer phrases first to avoid shorter prefix matches taking precedence
        sorted_words = sorted(words, key=len, reverse=True)
        escaped_words = [
            re.escape(w).replace(r"\ ", r"\s+") for w in sorted_words
        ]
        pattern_str = r"(?<!\w)(?:" + "|".join(escaped_words) + r")(?!\w)"
        _PATTERNS[category] = re.compile(pattern_str, flags=re.IGNORECASE)


def extract_stylometric_features(text: str) -> Dict[str, Any]:
    """
    Extract the exact 6 stylometric features from input text.

    Returns:
        dict containing:
            - 'features': dict of 6 frequencies per 100 words
            - 'counts': dict of raw match counts per category
            - 'word_count': total token count
            - 'sentence_count': estimated sentence count
            - 'avg_word_length': average character length of words
            - 'valid': whether text has sufficient length for analysis
            - 'error': error or warning message, if any
    """
    _init_lexicons()

    if not isinstance(text, str) or not text.strip():
        empty_feats = {name: 0.0 for name in FEATURE_NAMES}
        empty_counts = {name.split("_")[0]: 0 for name in FEATURE_NAMES}
        return {
            "features": empty_feats,
            "counts": empty_counts,
            "word_count": 0,
            "sentence_count": 0,
            "avg_word_length": 0.0,
            "valid": False,
            "error": "Input text is empty.",
        }

    tokens = TOKEN_PATTERN.findall(text.lower())
    word_count = len(tokens)

    if word_count == 0:
        empty_feats = {name: 0.0 for name in FEATURE_NAMES}
        empty_counts = {name.split("_")[0]: 0 for name in FEATURE_NAMES}
        return {
            "features": empty_feats,
            "counts": empty_counts,
            "word_count": 0,
            "sentence_count": 0,
            "avg_word_length": 0.0,
            "valid": False,
            "error": "No alphanumeric tokens found in input.",
        }

    # Sentences heuristic (split on whitespace following . ! ?)
    raw_sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()
    ]
    sentence_count = max(len(raw_sentences), 1)

    avg_wlen = sum(len(t) for t in tokens) / word_count

    extracted_features = {}
    raw_counts = {}

    for col in FEATURE_NAMES:
        category = col.split("_")[0]
        pattern = _PATTERNS.get(category)
        if pattern is not None:
            matches = pattern.findall(text)
            cnt = len(matches)
        else:
            cnt = 0

        freq = (cnt / word_count) * 100.0
        extracted_features[col] = float(freq)
        raw_counts[category] = cnt

    warning = None
    if word_count < 25:
        warning = f"Short document ({word_count} words). Stylometric frequencies are most reliable for texts > 50 words."

    return {
        "features": extracted_features,
        "counts": raw_counts,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_length": round(avg_wlen, 2),
        "valid": True,
        "error": warning,
    }


def get_feature_vector(features_dict: Dict[str, float]) -> List[float]:
    """Return features as an ordered 1D list matching model input expectations."""
    return [features_dict[col] for col in FEATURE_NAMES]
