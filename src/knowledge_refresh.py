"""Loopback-only retrieval refresh for the identity-bound operator importer."""
import asyncio
from fastapi import APIRouter, HTTPException, Request
from src.runtime_mode import generation_enabled
from src.contextual_retrieval.bm25_search import SmartBM25Search
router = APIRouter()


@router.post('/knowledge/refresh')
async def refresh(request: Request):
    if not request.client or request.client.host not in ('127.0.0.1', '::1'):
        raise HTTPException(403, 'Local operator transport required')
    if not generation_enabled():
        raise HTTPException(503, 'Generation disabled')
    service = getattr(request.app.state, 'orchestration_service', None)
    if service is None:
        raise HTTPException(503, 'Service unavailable')
    if not hasattr(request.app.state, 'knowledge_refresh_lock'):
        request.app.state.knowledge_refresh_lock = asyncio.Lock()
    async with request.app.state.knowledge_refresh_lock:
        shared = service.shared_bm25_search
        if shared is None:
            await service._prewarm_shared_bm25()
            shared = service.shared_bm25_search
            if shared is None:
                raise HTTPException(503, 'Knowledge refresh failed')
            with service._component_factory._cache_lock:
                service._component_factory._retriever_cache.clear()
        else:
            async with shared._refresh_lock:
                candidate = SmartBM25Search(shared.qdrant_url, config=shared._config)
                if not await candidate.initialize_index():
                    raise HTTPException(503, 'Knowledge refresh failed')
                # Publish the complete corpus without awaits between assignments.
                # Existing retrievers retain this shared object, so they see refreshes
                # even when a source edit leaves the collection point count unchanged.
                shared.bm25_index = candidate.bm25_index
                shared.chunk_mapping = candidate.chunk_mapping
                shared.last_collection_stats = candidate.last_collection_stats
        return {'status':'KNOWLEDGE_REFRESHED','chunks':len(shared.chunk_mapping)}
