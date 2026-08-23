"""
Text normalization & prompt enhancement utility for natural human-like TTS speech.

Adds natural breath pauses, expands abbreviations & numbers to words,
acronym phonemization, normalizes punctuation, and prepares raw text scripts
for expressive neural voice cloning.
"""
from __future__ import annotations

import re

# Abbreviation expansion map for smooth phoneme synthesis
ABBREVIATION_MAP = {
    r"\be\.g\.\b": "for example,",
    r"\bi\.e\.\b": "that is,",
    r"\betc\.\b": "et cetera.",
    r"\bvs\.\b": "versus",
    r"\bdr\.\b": "Doctor",
    r"\bmr\.\b": "Mister",
    r"\bmrs\.\b": "Missus",
    r"\bms\.\b": "Miss",
    r"\bprof\.\b": "Professor",
    r"\bst\.\b": "Saint",
    r"\bapprox\.\b": "approximately",
    r"\bmin\.\b": "minutes",
    r"\bmax\.\b": "maximum",
    r"\%": " percent",
    r"\&": " and ",
    r"\bno\.\b": "number",
    r"\bvol\.\b": "volume",
    r"\bdept\.\b": "department",
}

# Single digits & common numbers map for inline replacement
NUM_MAP = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
    "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen",
    "18": "eighteen", "19": "nineteen", "20": "twenty", "30": "thirty",
    "40": "forty", "50": "fifty", "60": "sixty", "70": "seventy",
    "80": "eighty", "90": "ninety", "100": "one hundred", "1000": "one thousand"
}

# Common uppercase acronyms to spell out with letter spacing (e.g. AI -> A. I.)
COMMON_ACRONYMS = [
    "AI", "API", "CPU", "GPU", "UI", "UX", "URL", "HTML", "CSS", "JS",
    "TTS", "DSP", "ID", "IQ", "EQ", "US", "UK", "USA", "EU", "OS", "DB", "RAM"
]


def _convert_number_to_words(match: re.Match) -> str:
    val_str = match.group(0)
    if val_str in NUM_MAP:
        return NUM_MAP[val_str]
    try:
        val = int(val_str)
        if 0 <= val <= 99:
            tens = (val // 10) * 10
            units = val % 10
            return f"{NUM_MAP[str(tens)]}-{NUM_MAP[str(units)]}"
    except ValueError:
        pass
    return val_str


def enhance_prompt_text(text: str) -> str:
    """
    Transforms raw unformatted script text into an expressive, natural-sounding prompt.
    - Expands contractions & abbreviations
    - Expands standalone numbers (1-99) into words
    - Formats acronyms with spaced letters for letter-by-letter reading (e.g., A. I.)
    - Inserts commas for natural breath pauses before conjunctions
    - Normalizes spacing and punctuation clutter
    - Preserves paralinguistic tags like [laughter], [sigh], [whisper], [pause]
    """
    if not text or not text.strip():
        return ""

    s = text.strip()

    # 0. Convert [pause] tag to breath ellipsis cue
    s = re.sub(r"\[pause\]", "...", s, flags=re.IGNORECASE)

    # 1. Expand abbreviations
    for pattern, repl in ABBREVIATION_MAP.items():
        s = re.sub(pattern, repl, s, flags=re.IGNORECASE)

    # 2. Format Acronyms (e.g., AI -> A. I., API -> A. P. I.)
    for acr in COMMON_ACRONYMS:
        pattern = rf"\b{acr}\b"
        spaced = ". ".join(list(acr)) + "."
        s = re.sub(pattern, spaced, s)

    # 3. Convert numbers to words for 0-99
    s = re.sub(r"\b\d{1,2}\b", _convert_number_to_words, s)

    # 4. Add comma before clause conjunctions if missing punctuation (e.g. "..., but ...")
    conjunctions = ["but", "however", "although", "because", "which", "while", "otherwise", "therefore", "nevertheless"]
    for conj in conjunctions:
        pattern = rf"(?<=[a-zA-Z0-9])\s+({conj}\b)"
        s = re.sub(pattern, r", \1", s, flags=re.IGNORECASE)

    # 5. Clean duplicate commas or periods while preserving ellipses
    s = re.sub(r",\s*,+", ",", s)
    s = re.sub(r"\.\s*\.\s*\.+", "...", s)  # keep triple dots (ellipses) intact
    s = re.sub(r"\?\s*\?+", "?", s)
    s = re.sub(r"!\s*!+", "!", s)

    # 6. Ensure trailing sentence punctuation if missing
    if s and s[-1] not in ".!?]\"'":
        s += "."

    # 7. Clean up redundant spaces
    s = re.sub(r"\s+", " ", s).strip()

    return s

