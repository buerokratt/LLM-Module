from types import SimpleNamespace
import json
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from src import widget_bot_bridge as bridge

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv('LLM_GENERATION_ENABLED','true')
    monkeypatch.setenv('WIDGET_LLM_ENABLED','true')
    monkeypatch.setattr(bridge,'testing_connection',lambda:'fixture-vault-uuid')
    app=FastAPI();app.include_router(bridge.router)
    class Service:
        calls=[]
        stored=False
        async def stream_orchestration_response(self, request):
            self.calls.append(request)
            for content in ['Library ', 'lends books.', 'END']:
                yield 'data: '+json.dumps({'payload':{'content':content}})+'\n\n'
            self.stored=True
    app.state.orchestration_service=Service()
    return TestClient(app), app.state.orchestration_service

BODY={'sender':'11111111-1111-4111-8111-111111111111','message':'What does the library provide?'}

def test_adapter_uses_testing_guarded_pipeline_and_rasa_shape(client):
    web,service=client;response=web.post('/webhooks/rest/webhook',json=BODY)
    assert response.status_code==200
    assert response.json()==[{'recipient_id':BODY['sender'],'text':'Library lends books.'}]
    request=service.calls[0]
    assert request.chatId==BODY['sender'] and request.environment=='testing' and request.connection_id=='fixture-vault-uuid'
    assert len(service.calls)==1 and service.stored

@pytest.mark.parametrize('flag',['WIDGET_LLM_ENABLED','LLM_GENERATION_ENABLED'])
def test_disabled_adapter_makes_no_provider_request(client,monkeypatch,flag):
    web,service=client;monkeypatch.setenv(flag,'false')
    assert web.post('/webhooks/rest/webhook',json=BODY).status_code==503
    assert service.calls==[]

def test_no_testing_connection_never_falls_back_to_production(client,monkeypatch):
    web,service=client;monkeypatch.setattr(bridge,'testing_connection',lambda:None)
    assert web.post('/webhooks/rest/webhook',json=BODY).status_code==503
    assert service.calls==[]

def test_invalid_chat_id_is_rejected(client):
    web,service=client
    assert web.post('/webhooks/rest/webhook',json={**BODY,'sender':'not-a-chat'}).status_code==422
    assert service.calls==[]

def test_truncated_stream_is_not_returned_as_success(client):
    web,service=client
    async def truncated(request):
        yield 'data: '+json.dumps({'payload':{'content':'unfinished'}})+'\n\n'
    service.stream_orchestration_response=truncated
    assert web.post('/webhooks/rest/webhook',json=BODY).status_code==502
