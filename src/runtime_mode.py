"""Explicit integration setting for the user's non-generating test deployment."""
import os


def generation_enabled() -> bool:
    value = os.getenv('LLM_GENERATION_ENABLED', 'true')
    if value not in ('true', 'false'):
        raise ValueError('LLM_GENERATION_ENABLED must be true or false')
    return value == 'true'
