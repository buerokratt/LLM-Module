"""Recover shared guardrails after transient configuration-service startup failures."""
import asyncio
from src.runtime_mode import generation_enabled

ENVIRONMENTS = ('production', 'testing')


def preload_missing(service, logger):
    """Keep ready adapters; publish new ones only after full initialization."""
    adapters = dict(getattr(service, 'shared_guardrails_adapters', {}))
    if not generation_enabled():
        return adapters
    for environment in ENVIRONMENTS:
        if environment in adapters:
            continue
        try:
            adapter = service._initialize_guardrails(environment=environment, connection_id=None)
            adapter._ensure_initialized()
        except Exception as exc:
            # Exception text may contain remote configuration/credential details.
            logger.warning(f'Shared guardrail preload unavailable for {environment} ({type(exc).__name__})')
            continue
        adapters[environment] = adapter
        logger.info(f'Shared guardrails ready for {environment}')
    return adapters


async def retry_preload(service, logger, initial_delay=5, max_delay=60):
    """One lifespan-owned worker, bounded backoff, no model generation or fallback."""
    delay = initial_delay
    while generation_enabled() and any(env not in service.shared_guardrails_adapters for env in ENVIRONMENTS):
        await asyncio.sleep(delay)
        if not generation_enabled():
            return
        # Synchronous DB/Vault/NeMo initialization must not block health or requests.
        attempt = asyncio.create_task(asyncio.to_thread(service._initialize_shared_guardrails_at_startup))
        try:
            adapters = await asyncio.shield(attempt)
        except asyncio.CancelledError:
            # A running thread cannot be cancelled. Drain it before service cleanup.
            await asyncio.gather(attempt, return_exceptions=True)
            raise
        except Exception as exc:
            logger.warning(f'Shared guardrail preload retry failed ({type(exc).__name__})')
        else:
            service.shared_guardrails_adapters = adapters
        delay = min(delay * 2, max_delay)
