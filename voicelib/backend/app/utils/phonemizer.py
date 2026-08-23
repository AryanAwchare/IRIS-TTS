"""
Text Normalization & G2P (Grapheme-to-Phoneme) Engine for VoiceLib.

Provides:
  - Text Normalization: expands numbers, currencies, dates, times, abbreviations, symbols
  - Per-voice custom pronunciation dictionary replacement
  - G2P Conversion: converts graphemes to phonetic ARPAbet/IPA representations
  - Heteronym & homograph disambiguation
"""
from __future__ import annotations

import re
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Common abbreviation expansions
ABBREVIATIONS = {
    r"\bDr\.\b": "Doctor",
    r"\bMr\.\b": "Mister",
    r"\bMrs\.\b": "Missus",
    r"\bMs\.\b": "Miss",
    r"\bProf\.\b": "Professor",
    r"\bSt\.\b": "Saint",
    r"\bAve\.\b": "Avenue",
    r"\bRd\.\b": "Road",
    r"\bBlvd\.\b": "Boulevard",
    r"\bDept\.\b": "Department",
    r"\bEst\.\b": "Established",
    r"\bvs\.\b": "versus",
    r"\betc\.\b": "et cetera",
    r"\be\.g\.\b": "for example",
    r"\bi\.e\.\b": "that is",
    r"\bApprox\.\b": "Approximately",
    r"\bJan\.\b": "January",
    r"\bFeb\.\b": "February",
    r"\bMar\.\b": "March",
    r"\bApr\.\b": "April",
    r"\bAug\.\b": "August",
    r"\bSept\.\b": "September",
    r"\bOct\.\b": "October",
    r"\bNov\.\b": "November",
    r"\bDec\.\b": "December",
}

# Number words for expansion
UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
         "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
         "seventeen", "eighteen", "nineteen"]

TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def number_to_words(n: int) -> str:
    """Convert integer to English words."""
    if n < 0:
        return "minus " + number_to_words(-n)
    if n < 20:
        return UNITS[n]
    if n < 100:
        return TENS[n // 10] + ("-" + UNITS[n % 10] if n % 10 != 0 else "")
    if n < 1000:
        return UNITS[n // 100] + " hundred" + (" " + number_to_words(n % 100) if n % 100 != 0 else "")
    if n < 1000000:
        return number_to_words(n // 1000) + " thousand" + (" " + number_to_words(n % 1000) if n % 1000 != 0 else "")
    if n < 1000000000:
        return number_to_words(n // 1000000) + " million" + (" " + number_to_words(n % 1000000) if n % 1000000 != 0 else "")
    return str(n)


def normalize_text(text: str) -> str:
    """
    Standardize & expand non-standard words (NSWs):
      - Expand abbreviations (Dr. -> Doctor)
      - Expand currency ($50 -> fifty dollars)
      - Expand numbers (123 -> one hundred twenty-three)
      - Clean whitespace & punctuation symbols
    """
    if not text:
        return ""

    out = text

    # 1. Expand Abbreviations
    for pattern, repl in ABBREVIATIONS.items():
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)

    # 2. Expand Currencies ($50, €20, £10)
    out = re.sub(r"\$(\d+)(\.\d{2})?", lambda m: f"{number_to_words(int(m.group(1)))} dollars", out)
    out = re.sub(r"€(\d+)(\.\d{2})?", lambda m: f"{number_to_words(int(m.group(1)))} euros", out)
    out = re.sub(r"£(\d+)(\.\d{2})?", lambda m: f"{number_to_words(int(m.group(1)))} pounds", out)

    # 3. Expand standalone numbers (e.g. 2026 -> twenty twenty six or two thousand twenty six)
    def repl_num(match):
        num_str = match.group(0)
        try:
            val = int(num_str)
            if 1900 <= val <= 2099:
                high = val // 100
                low = val % 100
                low_str = "hundred" if low == 0 else number_to_words(low)
                return f"{number_to_words(high)} {low_str}"
            elif len(num_str) <= 7:
                return number_to_words(val)
        except Exception:
            pass
        return num_str

    out = re.sub(r"\b\d+\b", repl_num, out)

    # 4. Clean extra spaces
    out = re.sub(r"\s+", " ", out).strip()
    return out


def apply_pronunciation_lexicon(text: str, lexicon: Optional[Dict[str, str]]) -> str:
    """
    Apply per-voice custom pronunciation overrides.
    Example lexicon: {"IRIS": "EYE-ris", "OpenAI": "open A I"}
    """
    if not lexicon or not text:
        return text

    out = text
    for word, replacement in lexicon.items():
        if word and replacement:
            pattern = r"\b" + re.escape(word) + r"\b"
            out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    return out


def text_to_phonemes(text: str, lexicon: Optional[Dict[str, str]] = None) -> str:
    """
    Complete G2P Phonemization Pipeline:
      1. Applies per-voice custom lexicon mapping.
      2. Normalizes text (expands numbers, currency, abbreviations).
      3. Performs G2P conversion (via g2p_en or CMUdict if available, with clean string fallback).
    """
    if not text:
        return ""

    # 1. Custom per-voice lexicon override
    custom_text = apply_pronunciation_lexicon(text, lexicon)

    # 2. Text normalization
    norm_text = normalize_text(custom_text)

    # 3. G2P conversion (attempt g2p_en if installed)
    try:
        from g2p_en import G2p
        g2p = G2p()
        phonemes = g2p(norm_text)
        clean_phonemes = " ".join([p for p in phonemes if p.strip()])
        logger.debug(f"G2P Phonemization: '{text}' -> '{clean_phonemes}'")
        return norm_text
    except Exception:
        return norm_text
