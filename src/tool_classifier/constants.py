"""Constants and configuration for tool classifier module."""


# ============================================================================
# Qdrant Vector Database Configuration
# ============================================================================

QDRANT_HOST = "qdrant"
"""Qdrant server hostname."""

QDRANT_PORT = 6333
"""Qdrant server port."""

QDRANT_TIMEOUT = 10.0
"""Qdrant HTTP client timeout in seconds."""


# ============================================================================
# Semantic Search Configuration
# ============================================================================

QDRANT_COLLECTION = "intent_collections"
"""Qdrant collection name for service intent search."""

SEMANTIC_SEARCH_TOP_K = 10
"""Number of top services to return from semantic search."""

SEMANTIC_SEARCH_THRESHOLD = 0.2
"""Minimum similarity score threshold for semantic search (0.0-1.0).
Lowered from 0.4 to handle broader queries."""


# ============================================================================
# Ruuter Service Configuration
# ============================================================================

RUUTER_BASE_URL = "http://ruuter-private:8086"
"""Base URL for Ruuter private service endpoints."""

RUUTER_SERVICE_BASE_URL = "http://ruuter:8086/services"
"""Base URL for Ruuter service endpoints (active services)."""

RUUTER_COMMON_SERVICE_BASE_URL = "http://ruuter-test:8086/common-services"
"""Base URL for Ruuter common service endpoints.
This is a placeholder test URL — replace with the real URL when available."""

RAG_SEARCH_RUUTER_PUBLIC = "http://ruuter-public:8086/rag-search"
"""Public Ruuter endpoint for RAG search service discovery."""

SERVICE_CALL_TIMEOUT = 10
"""Timeout in seconds for external service calls via Ruuter."""

SERVICE_DISCOVERY_TIMEOUT = 10.0
"""Timeout in seconds for service discovery calls."""

# ============================================================================
# Multi-Step Service (MCQ) Configuration
# ============================================================================

SERVICE_STEP_PREFIXES = ("#service,", "#common_service,")
"""Tuple of prefixes that identify a button-payload direct-step message.

When a user clicks an MCQ button, the widget sends the button's payload string
as the next user message. These prefixes identify such machine-generated
commands so the orchestrator can bypass NLU and route directly to the
step endpoint.

Examples:
    "#service, /POST/services/active/application_mcq_step_passport"
    "#common_service, /POST/common/some_step"
"""


# ============================================================================
# Service Workflow Thresholds
# ============================================================================

MAX_SERVICES_FOR_LLM_CONTEXT = 50
"""Maximum number of services to send to LLM without semantic filtering.
If service count exceeds this, semantic search is used to filter to top-K."""

SERVICE_COUNT_THRESHOLD = 10
"""Threshold for triggering semantic search. If service count > this value,
semantic search is used instead of sending all services to LLM."""


# ============================================================================
# Hybrid Search Classification Thresholds
# ============================================================================

HYBRID_SEARCH_TOP_K = 5
"""Number of top results from hybrid search for service identification."""

DENSE_SEARCH_TOP_K = 3
"""Number of top results from dense-only search for relevance scoring."""

DENSE_MIN_THRESHOLD = 0.5
"""Minimum dense cosine similarity to consider a result as a potential match.
Below this → skip SERVICE entirely, go to CONTEXT/RAG.
Note: Multilingual embeddings (Estonian/short queries) typically yield
lower cosine scores (0.25-0.40) than English. Tune based on observed scores."""

DENSE_HIGH_CONFIDENCE_THRESHOLD = 0.55
"""Dense cosine similarity for high-confidence service classification.
Above this AND score gap is large → SERVICE without LLM confirmation."""

DENSE_SCORE_GAP_THRESHOLD = 0.05
"""Cosine score gap (top - second) for high-confidence classification.
Ensures the top result is significantly better than the runner-up."""


# ============================================================================
# API Tool Collection Search Configuration
# ============================================================================

API_TOOL_COLLECTION = "api_tool_collection"
"""Qdrant collection name for API endpoint semantic search."""

API_TOOL_SEARCH_TOP_K = 5
"""Number of top endpoints to return from API tool semantic search."""

API_TOOL_MIN_THRESHOLD = 0.40
"""Minimum dense cosine similarity to consider a result as an API tool match.
Below this → no API tool matched, fall through to other workflows."""

API_TOOL_HIGH_CONFIDENCE_THRESHOLD = 0.60
"""Dense cosine similarity for high-confidence API tool match.
Above this AND score gap is large → route to API Tool Calling without further LLM disambiguation."""

API_TOOL_SCORE_GAP_THRESHOLD = 0.05
"""Cosine score gap (top - second) for high-confidence API tool classification."""

API_TOOL_INTENT_SWITCH_THRESHOLD = 0.50
"""Minimum cosine required to abandon an active session and switch intent.
Higher than API_TOOL_MIN_THRESHOLD (0.40).
Only a clear, unambiguous new query (cosine >= 0.50) should override a session."""


# ============================================================================
# Agentic Loop — Continuation Threshold
# ============================================================================

MULTI_API_MAX_TURNS = 9
"""Hard cap on total turns for the multi-endpoint agentic loop (MultiEndpointAgenticLoop).

The per-session limit is ``min(3 * num_endpoints, MULTI_API_MAX_TURNS)``.
``MULTI_API_MAX_ENDPOINTS`` independently caps ``num_endpoints`` at 3, so the
combined ceiling is ``min(3 * 3, 9) = 9`` turns — up to 3 turns per endpoint
at maximum endpoint count.
"""

CONTINUATION_TURN = 3
"""1-based turn count (after increment) at which the loop asks the user whether
to continue collecting parameters or fall back to the RAG workflow.
Only triggers when required params are still missing at exactly this turn.

The turn counter is incremented on every run_turn() call, including the
initial call that generates the bot's opening question (before the user
speaks). With CONTINUATION_TURN=3 the conversation looks like:

  run_turn #1 (turn 0→1): initial question — "Which country and date?"
  run_turn #2 (turn 1→2): user gives partial answer — bot asks follow-up
  run_turn #3 (turn 2→3): user doesn't answer properly → CONTINUATION CHECK
"""

MULTI_INTENT_CONTINUATION_TURN = 4
"""1-based turn count at which the multi-endpoint loop asks the user whether to
continue collecting parameters when **multiple** intents (endpoints) are active.
Fixed at 4 regardless of how many endpoints are active — if required params are
still missing at exactly this turn, the user is asked whether to keep going.
"""

MULTI_INTENT_MAX_TURNS = 6
"""Hard cap on total turns for the multi-endpoint agentic loop when multiple
intents are active. When the internal turn counter reaches or exceeds this value
(i.e., when `turn_count >= MULTI_INTENT_MAX_TURNS`, such as when attempting
`turn_count == 6`), the loop falls back to the RAG workflow regardless of how
many required params are still missing.
"""

CONTINUATION_QUESTION = (
    "I still need a bit more information, but we've been at this for a while. "
    "Would you like to keep going and answer a few more questions "
    "(yes / no)"
)
"""Yes/no question shown to the user when the continuation threshold is reached."""

CONTINUATION_QUESTION_ET = (
    "Mul on vaja veel natuke lisateavet, kuid oleme selle kallal juba mõnda aega töötanud. "
    "Kas soovite jätkata ja vastata veel mõnele küsimusele? (jah / ei)"
)
"""Estonian version of the continuation question."""

CONTINUATION_QUESTION_RU = (
    "Мне нужно ещё немного информации, но мы уже некоторое время занимаемся этим. "
    "Хотите ли вы продолжить и ответить ещё на несколько вопросов? (да / нет)"
)
"""Russian version of the continuation question."""

# ============================================================================
# API Caller Configuration
# ============================================================================

API_CALL_TIMEOUT = 10
"""Default timeout in seconds for external API calls made via APICaller."""

# Circuit breaker state literals
CB_STATE_CLOSED = "CLOSED"
"""Circuit breaker is CLOSED: routes all requests normally."""

CB_STATE_OPEN = "OPEN"
"""Circuit breaker is OPEN: rejects all requests immediately (cooldown active)."""

CB_STATE_HALF_OPEN = "HALF_OPEN"
"""Circuit breaker is HALF_OPEN: allows one probe request to test recovery."""

CIRCUIT_BREAKER_FAILURE_THRESHOLD = 3
"""Number of consecutive server/network failures before the circuit breaker opens."""

CIRCUIT_BREAKER_COOLDOWN_SECONDS = 60.0
"""Seconds the circuit breaker stays OPEN before transitioning to HALF_OPEN."""

# User-facing error messages for API call failures (multilingual: et / ru / en)
SERVICE_UNAVAILABLE_MESSAGES = {
    "et": "Teenus on ajutiselt kättesaamatu. Palun proovige hiljem uuesti.",
    "ru": "Сервис временно недоступен. Пожалуйста, попробуйте позже.",
    "en": "The service is temporarily unavailable. Please try again later.",
}
"""Friendly message returned on 5xx server errors."""

SERVICE_TIMEOUT_MESSAGES = {
    "et": "Teenuse päring aegus. Palun proovige mõne hetke pärast uuesti.",
    "ru": "Запрос к сервису истёк по таймауту. Пожалуйста, попробуйте снова через несколько секунд.",
    "en": "The service request timed out. Please try again in a moment.",
}
"""Friendly message returned on timeout or network errors."""

CIRCUIT_BREAKER_OPEN_MESSAGES = {
    "et": "Teenus on praegu kättesaamatu korduvate vigade tõttu. Palun proovige hiljem uuesti.",
    "ru": "Сервис в данный момент недоступен из-за повторяющихся ошибок. Пожалуйста, попробуйте позже.",
    "en": "The service is currently unavailable due to repeated failures. Please try again later.",
}
"""Friendly message returned when the circuit breaker is open."""

REDIRECT_NOT_FOLLOWED_MESSAGES = {
    "et": "Teenus tagastas ümbersuunamise, mida ei järgitud (HTTP {status_code}): {location}",
    "ru": "Сервис вернул перенаправление, которое не было выполнено (HTTP {status_code}): {location}",
    "en": "The service returned an unresolved redirect (HTTP {status_code}): {location}",
}
"""Friendly message returned when an HTTP 3xx redirect could not be followed."""

CLIENT_ERROR_MESSAGES = {
    "et": "Teie päringut ei saanud töödelda. Palun kontrollige sisestatud andmeid ja proovige uuesti.",
    "ru": "Ваш запрос не удалось обработать. Пожалуйста, проверьте введённые данные и повторите попытку.",
    "en": "Your request could not be processed. Please check the provided information and try again.",
}
"""Friendly message returned on 4xx client errors from external API calls."""


# ============================================================================
# Multi-Intent (Parallel Multi-API) Configuration
# ============================================================================

MULTI_API_MAX_ENDPOINTS = 3
"""Maximum number of parallel endpoints allowed per multi-intent query.
Caps the number of sub-queries the IntentDecomposer may produce and the
number of concurrent API calls MultiAPICaller may execute."""

MULTI_API_BATCH_TIMEOUT = 30
"""Total wall-clock timeout in seconds for all concurrent API calls in a batch.
Exceeds the per-call API_CALL_TIMEOUT (10 s) to allow parallel requests time to
complete without the batch being cancelled prematurely."""

MULTI_API_PARTIAL_FAILURE_MESSAGES = {
    "et": "Mõned teenusekutsed ebaõnnestusid. Osad tulemused võivad puududa.",
    "ru": "Некоторые запросы к сервисам завершились неудачно. Часть результатов может отсутствовать.",
    "en": "Some service calls failed. Partial results may be missing.",
}
"""Friendly message returned when a batch of API calls has partial failures."""


# ============================================================================
# ATC Response Cache Configuration
# ============================================================================

ATC_CACHE_KEY_PREFIX = "atc:cache"
"""Redis key prefix for L1 exact-match response cache entries.
Full key format: atc:cache:{chat_id}:{api_name}:{param_hash}"""

ATC_LAST_CALL_KEY_PREFIX = "atc:last"
"""Redis key prefix for L2 last-call context entries.
Full key format: atc:last:{chat_id}"""

ATC_CACHE_DEFAULT_TTL_SECONDS = 1800
"""Default L1 cache TTL in seconds (30 minutes).
Applied when an endpoint does not define a per-endpoint cache_ttl_seconds override."""

ATC_LAST_CALL_TTL_SECONDS = 1800
"""L2 last-call context TTL in seconds (30 minutes).
Matches the session TTL so cached context never outlives the session."""
