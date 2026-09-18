# Multilingual message dictionaries
OUT_OF_SCOPE_MESSAGES = {
    "et": "Vabandust, kuid mul pole piisavalt konteksti, et teie küsimusele vastata. Palun püüdke ümber sõnastada või lisage rohkem üksikasju.",
    "ru": "Извините, но у меня недостаточно контекста для ответа на ваш вопрос. Пожалуйста, попробуйте переформулировать или предоставить больше деталей.",
    "en": "I apologize, but I'm unable to provide a complete response because the available context does not sufficiently cover your request. Please try rephrasing or providing more details.",
}

TECHNICAL_ISSUE_MESSAGES = {
    "et": "Tehniline probleem vastuse genereerimisel\nVabandust, kuid ma ei saa praegu vastust genereerida ajutise tehnilise probleemi tõttu. Palun proovige mõne hetke pärast uuesti.",
    "ru": "Техническая проблема при генерации ответа\nИзвините, в настоящее время я не могу сгенерировать ответ из-за временной технической проблемы. Пожалуйста, попробуйте еще раз через мгновение.",
    "en": "Technical issue with response generation\nI apologize, but I'm currently unable to generate a response due to a temporary technical issue. Please try again in a moment.",
}

INPUT_GUARDRAIL_VIOLATION_MESSAGES = {
    "et": "Vabandust, kuid ma ei saa selle taotlusega aidata, kuna see rikub meie kasutustingimusi.",
    "ru": "Извините, но я не могу помочь с этим запросом, так как он нарушает нашу политику использования.",
    "en": "I apologize, but I'm unable to assist with that request as it violates our usage policies.",
}

OUTPUT_GUARDRAIL_VIOLATION_MESSAGES = {
    "et": "Vabandust, kuid ma ei saa vastust anda, kuna see võib rikkuda meie kasutustingimusi.",
    "ru": "Извините, но я не могу предоставить ответ, так как он может нарушить нашу политику использования.",
    "en": "I apologize, but I'm unable to provide a response as it may violate our usage policies.",
}

# Query validation messages - single generic message for all rejection types
# (empty queries, special characters only, too short, repetitive characters)
QUERY_VALIDATION_FAILED_MESSAGES = {
    "et": "Palun esitage kehtiv küsimus või sõnum, et ma saaksin teid aidata.",
    "en": "Please provide a valid question or reply so I can assist you.",
    "ru": "Пожалуйста, введите корректный вопрос или сообщение, чтобы я мог вам помочь.",
}

# Legacy constants for backward compatibility (English defaults)
OUT_OF_SCOPE_MESSAGE = OUT_OF_SCOPE_MESSAGES["en"]
TECHNICAL_ISSUE_MESSAGE = TECHNICAL_ISSUE_MESSAGES["en"]
INPUT_GUARDRAIL_VIOLATION_MESSAGE = INPUT_GUARDRAIL_VIOLATION_MESSAGES["en"]
OUTPUT_GUARDRAIL_VIOLATION_MESSAGE = OUTPUT_GUARDRAIL_VIOLATION_MESSAGES["en"]

UNKNOWN_SOURCE = "Unknown source"

GUARDRAILS_BLOCKED_PHRASES = [
    "i'm sorry, i can't respond to that",
    "i cannot respond to that",
    "i cannot help with that",
    "this is against policy",
]

# Streaming configuration
STREAMING_ALLOWED_ENVS = {"production", "testing"}
TEST_DEPLOYMENT_ENVIRONMENT = "testing"
PRODUCTION_DEPLOYMENT_ENVIRONMENT = "production"

# Stream limit error messages
STREAM_TIMEOUT_MESSAGE = (
    "I apologize, but generating your response is taking longer than expected. "
    "Please try asking your question in a simpler way or break it into smaller parts."
)

STREAM_TOKEN_LIMIT_MESSAGE = (
    "I apologize, but I've reached the maximum response length for this question. "
    "The answer provided above covers the main points, but some details may have been abbreviated. "
    "Please feel free to ask follow-up questions for more information."
)

STREAM_SIZE_LIMIT_MESSAGE = (
    "I apologize, but your request is too large to process. "
    "Please shorten your message or reduce the conversation history and try again."
)

STREAM_CAPACITY_EXCEEDED_MESSAGE = (
    "I apologize, but our service is currently at capacity. "
    "Please wait a moment and try again. Thank you for your patience."
)

STREAM_USER_LIMIT_EXCEEDED_MESSAGE = (
    "I apologize, but you have reached the maximum number of concurrent conversations. "
    "Please wait for your existing conversations to complete before starting a new one."
)

# Rate limiting error messages
RATE_LIMIT_REQUESTS_EXCEEDED_MESSAGE = (
    "I apologize, but you've made too many requests in a short time. "
    "Please wait a moment before trying again."
)

RATE_LIMIT_TOKENS_EXCEEDED_MESSAGE = (
    "I apologize, but you're sending requests too quickly. "
    "Please slow down and try again in a few seconds."
)

# Validation error messages
VALIDATION_MESSAGE_TOO_SHORT = "Please provide a message with at least a few characters so I can understand your request."

VALIDATION_MESSAGE_TOO_LONG = (
    "Your message is too long. Please shorten it and try again."
)

VALIDATION_MESSAGE_INVALID_FORMAT = (
    "Please provide a valid message without special formatting."
)

VALIDATION_MESSAGE_GENERIC = "Please provide a valid message for your request."

VALIDATION_CONVERSATION_HISTORY_ERROR = (
    "There was an issue with the conversation history format. Please try again."
)

VALIDATION_REQUEST_TOO_LARGE = "Your request is too large. Please reduce the message size or conversation history and try again."

VALIDATION_REQUIRED_FIELDS_MISSING = "Required information is missing from your request. Please ensure all required fields are provided."

VALIDATION_GENERIC_ERROR = "I apologize, but I couldn't process your request. Please check your input and try again."


# Helper function to get localized messages
def get_localized_message(message_dict: dict, language_code: str = "et") -> str:
    """
    Get message in the specified language, fallback to Estonian.

    Args:
        message_dict: Dictionary with language codes as keys
        language_code: Language code ('et', 'ru', 'en')

    Returns:
        Localized message string
    """
    return message_dict.get(language_code, message_dict.get("et", ""))


# Service endpoints
RAG_SEARCH_RESQL = "http://resql:8082/rag-search"
RAG_SEARCH_RUUTER_PUBLIC = "http://ruuter-public:8086/rag-search"
RAG_SEARCH_RUUTER_PRIVATE = "http://ruuter-private:8088/rag-search"

# Custom Prompt Configuration
RUUTER_PROMPT_CONFIG_ENDPOINT = (
    "http://ruuter-public:8086/rag-search/llm-connections/prompts/get-prompt"
)
PROMPT_CONFIG_CACHE_TTL = 300  # 5 minutes cache
