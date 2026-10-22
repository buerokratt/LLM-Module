"""Constants for greeting responses in multiple languages."""

from typing import Dict

# Estonian greeting responses
GREETINGS_ET: Dict[str, str] = {
    "hello": "Tere! Kuidas ma saan sind aidata?",
    "goodbye": "Nägemist! Head päeva!",
    "thanks": "Palun! Kui on veel küsimusi, küsi julgelt.",
    "casual": "Tere! Mida ma saan sinu jaoks teha?",
}

# English greeting responses
GREETINGS_EN: Dict[str, str] = {
    "hello": "Hello! How can I help you?",
    "goodbye": "Goodbye! Have a great day!",
    "thanks": "You're welcome! Feel free to ask if you have more questions.",
    "casual": "Hey! What can I do for you?",
}

# Russian greeting responses
GREETINGS_RU: Dict[str, str] = {
    "hello": "Здравствуйте! Чем я могу вам помочь?",
    "goodbye": "До свидания! Хорошего дня!",
    "thanks": "Пожалуйста! Если есть ещё вопросы, спрашивайте.",
    "casual": "Привет! Что я могу для вас сделать?",
}

# Language-specific greeting mappings
GREETINGS_BY_LANGUAGE: Dict[str, Dict[str, str]] = {
    "et": GREETINGS_ET,
    "en": GREETINGS_EN,
    "ru": GREETINGS_RU,
}


def get_greeting_response(greeting_type: str = "hello", language: str = "et") -> str:
    """
    Get a greeting response for a specific type and language.

    Args:
        greeting_type: Type of greeting (hello, goodbye, thanks, casual)
        language: Language code (et, en)

    Returns:
        Greeting message in the specified language
    """
    language_greetings = GREETINGS_BY_LANGUAGE.get(language, GREETINGS_EN)
    return language_greetings.get(greeting_type, language_greetings["hello"])
