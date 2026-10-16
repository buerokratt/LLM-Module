import asyncio
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock
import httpx
from fastapi import FastAPI
from src import knowledge_refresh as module


def test_only_loopback_can_refresh_and_changed_corpus_is_published(monkeypatch):
    monkeypatch.setenv('LLM_GENERATION_ENABLED','true')
    shared=SimpleNamespace(_refresh_lock=asyncio.Lock(),qdrant_url='http://fixture',_config=None,bm25_index='old',chunk_mapping={0:'old'},last_collection_stats={})
    candidate=SimpleNamespace(initialize_index=AsyncMock(return_value=True),bm25_index='new',chunk_mapping={0:'new-same-count'},last_collection_stats={'count':1})
    factory=Mock(return_value=candidate);monkeypatch.setattr(module,'SmartBM25Search',factory)
    app=FastAPI();app.include_router(module.router);app.state.orchestration_service=SimpleNamespace(shared_bm25_search=shared)
    async def run(host):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app,client=(host,5000)),base_url='http://fixture') as client:
            return await client.post('/knowledge/refresh')
    assert asyncio.run(run('10.42.0.1')).status_code==403
    factory.assert_not_called()
    response=asyncio.run(run('127.0.0.1'))
    assert response.status_code==200 and response.json()=={'status':'KNOWLEDGE_REFRESHED','chunks':1}
    assert shared.chunk_mapping=={0:'new-same-count'} and shared.bm25_index=='new'
    candidate.initialize_index.return_value=False
    candidate.chunk_mapping={0:'broken'}
    assert asyncio.run(run('127.0.0.1')).status_code==503
    assert shared.chunk_mapping=={0:'new-same-count'}
    monkeypatch.setenv('LLM_GENERATION_ENABLED','false')
    assert asyncio.run(run('127.0.0.1')).status_code==503
