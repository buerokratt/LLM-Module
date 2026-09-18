"""Language detection utility for multilingual support.

Detects Estonian, Russian, and English based on character patterns and common words.
"""

import re
from typing import Literal
from src.loki_logger import LokiLogger

# Initialize Loki logger
logger = LokiLogger(service_name="language-detector")

LanguageCode = Literal["et", "ru", "en"]


def detect_language(text: str) -> LanguageCode:
    """
    Detect language from input text.

    Detection Strategy:
    1. Check for Cyrillic characters (Russian)
    2. Check for Estonian-specific characters
    3. Check for Estonian common words
    4. Default to English

    Args:
        text: Input text to analyze

    Returns:
        Language code: 'et' (Estonian), 'ru' (Russian), 'en' (English)

    Examples:
        >>> detect_language("Mis on sünnitoetus?")
        'et'
        >>> detect_language("Что такое пособие?")
        'ru'
        >>> detect_language("What is the benefit?")
        'en'
    """
    if not text or not text.strip():
        logger.warning(
            "Empty text provided for language detection, defaulting to English"
        )
        return "en"

    text_sample = text.strip()[:500]  # Use first 500 chars for detection

    # Check for Cyrillic characters (Russian) - use percentage-based detection
    cyrillic_count = len(re.findall(r"[а-яА-ЯёЁ]", text_sample))
    total_alpha = len(re.findall(r"[a-zA-Zа-яА-ЯёЁõäöüšžÕÄÖÜŠŽ]", text_sample))

    if (
        total_alpha > 0 and cyrillic_count / total_alpha > 0.25
    ):  # 25% Cyrillic threshold
        logger.debug(
            f"Detected Russian (Cyrillic: {cyrillic_count}/{total_alpha} = {cyrillic_count / total_alpha:.1%})"
        )
        return "ru"

    # Check for Estonian-specific characters (õ, ä, ö, ü, š, ž)
    estonian_chars = re.findall(r"[õäöüšž]", text_sample, re.IGNORECASE)
    if len(estonian_chars) > 0:
        logger.debug(f"Detected Estonian (special chars: {len(estonian_chars)})")
        return "et"

    # Check for Estonian common words - use distinctive markers to avoid English false positives
    estonian_markers = [
        "kuidas",
        "miks",
        "kus",
        "millal",
        "kes",
        "võib",
        "olen",
        "oled",
        "seda",
        "jah",
        "või",
        "ning",
        "siis",
        "veel",
        "aga",
        "kuid",
        "nii",
        "nagu",
        "oli",
        "mis",
        # Estonian greeting words
        "tere",
        "hei",
        "tervist",
        "tänan",
        "aitäh",
        "nägemist",
        "moi",
    ]

    # Tokenize and check for Estonian markers
    words = re.findall(r"\b\w+\b", text_sample.lower())
    estonian_word_count = sum(1 for word in words if word in estonian_markers)

    # Scale threshold based on text length for better accuracy
    threshold = 1 if len(words) < 10 else 2
    if estonian_word_count >= threshold:
        logger.debug(
            f"Detected Estonian (marker words: {estonian_word_count}/{len(words)}, threshold: {threshold})"
        )
        return "et"

    # Default to English
    logger.debug("Detected English (default)")
    return "en"


def get_language_name(language_code: LanguageCode) -> str:
    """
    Get human-readable language name from code.

    Args:
        language_code: ISO 639-1 language code

    Returns:
        Full language name
    """
    language_names = {"et": "Estonian", "ru": "Russian", "en": "English"}
    return language_names.get(language_code, "Unknown")
