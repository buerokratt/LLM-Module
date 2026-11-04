"""Notification relay preserves streamed data and reports incomplete streams."""
import json
import unittest

from src.utils.notification_publisher import StreamRelay


class RecordingPublisher:
    def __init__(self):
        self.events = []

    async def publish(self, chat_id, event_type, payload):
        self.events.append((chat_id, event_type, payload))


def frame(content, **extra):
    return 'data: ' + json.dumps({'payload': {'content': content, **extra}}) + '\n\n'


async def chunks(*values):
    for value in values:
        yield value


class RelayTests(unittest.IsolatedAsyncioTestCase):
    async def test_fragmented_stream_preserves_text_and_buttons(self):
        publisher = RecordingPublisher()
        data = frame('Hello', buttons=[{'title': 'Yes', 'payload': 'yes'}]) + frame(' world') + frame('END')
        await StreamRelay(publisher, 'chat', 'request').run(chunks(data[:20], data[20:]))
        self.assertEqual([event[1] for event in publisher.events], ['stream_start', 'stream_chunk', 'stream_chunk', 'stream_complete'])
        self.assertEqual(publisher.events[1][2]['content'], 'Hello')
        self.assertEqual(publisher.events[1][2]['buttons'][0]['payload'], 'yes')
        self.assertEqual(publisher.events[2][2]['content'], ' world')
        self.assertTrue(all(event[2]['requestId'] == 'request' for event in publisher.events))

    async def test_missing_end_emits_error_not_complete(self):
        publisher = RecordingPublisher()
        await StreamRelay(publisher, 'chat', 'request').run(chunks(frame('partial')))
        self.assertEqual([event[1] for event in publisher.events], ['stream_start', 'stream_chunk', 'stream_error'])

    async def test_heartbeat_is_not_delivered_as_text(self):
        publisher = RecordingPublisher()
        await StreamRelay(publisher, 'chat', 'request').run(chunks(': heartbeat\n\n' + frame('END')))
        self.assertEqual([event[1] for event in publisher.events], ['stream_start', 'stream_complete'])

    async def test_utf8_character_split_across_chunks(self):
        publisher = RecordingPublisher()
        data = ('data: {"payload":{"content":"õ"}}\n\n' + frame('END')).encode()
        split_at = data.index('õ'.encode()) + 1
        await StreamRelay(publisher, 'chat', 'request').run(chunks(data[:split_at], data[split_at:]))
        self.assertEqual(publisher.events[1][2]['content'], 'õ')

    async def test_failed_delivery_reports_stream_error(self):
        class FailingPublisher(RecordingPublisher):
            async def publish(self, chat_id, event_type, payload):
                if event_type == 'stream_chunk':
                    raise RuntimeError('delivery failed')
                await super().publish(chat_id, event_type, payload)
        publisher = FailingPublisher()
        await StreamRelay(publisher, 'chat', 'request').run(chunks(frame('hello'), frame('END')))
        self.assertEqual([event[1] for event in publisher.events], ['stream_start', 'stream_error'])


class NotifyEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from unittest.mock import patch
        import httpx
        from src.llm_orchestration_service_api import app
        self.app = app
        self.app.state.notification_streams = {}
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test')
        self.env = patch.dict('os.environ', {'RAG_SEARCH_NOTIFICATIONS_NODE_PRIVATE': 'http://node:3001'})
        self.env.start()
        self.payload = {'chatId': '4cc1ee4f-765d-4f92-94de-1e5ab4caa918', 'message': 'hello', 'authorId': 'user', 'conversationHistory': [], 'url': 'test', 'environment': 'production'}

    async def asyncTearDown(self):
        import asyncio
        tasks = [entry['task'] for entry in self.app.state.notification_streams.values()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.client.aclose()
        self.env.stop()

    async def test_rejects_missing_cookie_and_non_uuid_chat(self):
        response = await self.client.post('/orchestrate/stream/notify', json=self.payload)
        self.assertEqual(response.status_code, 401)
        response = await self.client.post('/orchestrate/stream/notify', headers={'Cookie': 'customJwtCookie=test'}, json={**self.payload, 'chatId': 'channel-test'})
        self.assertEqual(response.status_code, 422)

    async def test_returns_202_and_relays_existing_stream(self):
        import asyncio
        from unittest.mock import patch
        from fastapi.responses import StreamingResponse
        publisher = RecordingPublisher()
        async def stream(*args):
            return StreamingResponse(chunks(frame('hello'), frame('END')))
        with patch('src.llm_orchestration_service_api.stream_orchestrated_response', stream), patch('src.llm_orchestration_service_api.NotificationPublisher', return_value=publisher):
            response = await self.client.post('/orchestrate/stream/notify', headers={'Cookie': 'customJwtCookie=test'}, json=self.payload)
            self.assertEqual(response.status_code, 202)
            request_id = response.json()['requestId']
            task = self.app.state.notification_streams[request_id]['task']
            await asyncio.wait_for(task, timeout=2)
        self.assertEqual([event[1] for event in publisher.events], ['stream_start', 'stream_chunk', 'stream_complete'])
        self.assertEqual(publisher.events[1][2]['requestId'], request_id)

    async def test_rate_limit_is_returned_before_task_is_started(self):
        from unittest.mock import patch
        from fastapi.responses import StreamingResponse
        async def stream(*args):
            return StreamingResponse(chunks(frame('limited'), frame('END')), status_code=429)
        with patch('src.llm_orchestration_service_api.stream_orchestrated_response', stream):
            response = await self.client.post('/orchestrate/stream/notify', headers={'Cookie': 'customJwtCookie=test'}, json=self.payload)
        self.assertEqual(response.status_code, 429)
        self.assertEqual(self.app.state.notification_streams, {})

    async def test_cancel_by_request_id_without_cookie(self):
        import asyncio
        from unittest.mock import patch
        from fastapi.responses import StreamingResponse
        blocker = asyncio.Event()
        async def long_stream():
            await blocker.wait()
            yield frame('END')
        async def stream(*args):
            return StreamingResponse(long_stream())
        publisher = RecordingPublisher()
        with patch('src.llm_orchestration_service_api.stream_orchestrated_response', stream), patch('src.llm_orchestration_service_api.NotificationPublisher', return_value=publisher):
            response = await self.client.post('/orchestrate/stream/notify', headers={'Cookie': 'customJwtCookie=test'}, json=self.payload)
            request_id = response.json()['requestId']
            url = f'/orchestrate/stream/notify/{request_id}/cancel'
            self.assertEqual((await self.client.post(url)).status_code, 204)
            self.assertEqual((await self.client.post(url)).status_code, 404)
