"""Opt-in, internal Rasa response adapter for the isolated widget acceptance path."""
import asyncio
import os
import json
from uuid import UUID
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from src.runtime_mode import generation_enabled
from models.request_models import OrchestrationRequest

router = APIRouter()

class WidgetBotRequest(BaseModel):
    sender: str
    message: str = Field(min_length=1, max_length=8000)

    @field_validator('sender')
    @classmethod
    def chat_uuid(cls, value):
        UUID(value)
        return value


def testing_connection():
    from src.utils.connection_id_fetcher import get_connection_id_fetcher
    return get_connection_id_fetcher().fetch_vault_uuid_sync('testing')


@router.post('/webhooks/rest/webhook')
async def widget_answer(body: WidgetBotRequest, request: Request):
    if os.getenv('WIDGET_LLM_ENABLED', 'false') != 'true' or not generation_enabled():
        raise HTTPException(status_code=503, detail='Widget LLM route disabled')
    service = getattr(request.app.state, 'orchestration_service', None)
    if service is None:
        raise HTTPException(status_code=503, detail='Orchestrator not ready')
    connection = await asyncio.to_thread(testing_connection)
    if not connection:
        raise HTTPException(status_code=503, detail='Testing connection required')
    payload = OrchestrationRequest(
        chatId=body.sender, message=body.message, authorId='widget-'+body.sender,
        conversationHistory=[], url='central-widget-test', environment='testing',
        connection_id=connection,
    )
    # Drain the proven guarded stream through EOF: storage runs after END.
    # The existing Rasa caller expects one final answer, not an SSE response.
    parts = []
    terminal = False
    buttons = None
    try:
        async with asyncio.timeout(80):
            async for frame in service.stream_orchestration_response(payload):
                for line in frame.splitlines():
                    if not line.startswith('data: '):
                        continue
                    event = json.loads(line[6:]).get('payload', {})
                    content = event.get('content')
                    if content == 'END':
                        terminal = True
                    elif isinstance(content, str):
                        parts.append(content)
                    if event.get('buttons'):
                        buttons = event['buttons']
    except TimeoutError:
        raise HTTPException(status_code=504, detail='Widget response timed out') from None
    if not terminal or not parts:
        raise HTTPException(status_code=502, detail='Incomplete widget response')
    answer = {'recipient_id':body.sender, 'text':''.join(parts)}
    if buttons:
        answer['buttons'] = buttons
    return [answer]
