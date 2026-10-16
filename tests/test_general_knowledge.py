import asyncio
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock
import pytest
from src.response_generator import general_knowledge as general


def test_mode_is_explicit_and_testing_only(monkeypatch):
    monkeypatch.setenv('LLM_GENERATION_ENABLED', 'true')
    monkeypatch.delenv('ALLOW_GENERAL_KNOWLEDGE', raising=False)
    assert not general.enabled('testing')
    monkeypatch.setenv('ALLOW_GENERAL_KNOWLEDGE', 'true')
    assert general.enabled('testing') and not general.enabled('production')
    monkeypatch.setenv('LLM_GENERATION_ENABLED', 'false')
    assert not general.enabled('testing')


@pytest.mark.parametrize('lang', ['et', 'en', 'ru'])
def test_general_answer_has_no_fabricated_source_and_is_labelled(monkeypatch, lang):
    predictor = Mock(return_value=SimpleNamespace(answer='Plain general answer'))
    monkeypatch.setattr(general.dspy, 'Predict', Mock(return_value=predictor))
    monkeypatch.setattr(general.dspy, 'settings', SimpleNamespace(lm=None))
    agent = general.GeneralKnowledgeAgent(lang)
    result = agent.forward(question='What is photosynthesis?')
    assert result['answer'] == general.label(lang)+'Plain general answer'
    assert not result['questionOutOfLLMScope'] and 'http' not in result['answer']


@pytest.mark.parametrize('chunks,out_of_scope', [([], False), ([{'text':'irrelevant','meta':{'source':'wrong'}}], True)])
def test_stream_general_fallback_uses_output_rails_and_single_storage(monkeypatch, chunks, out_of_scope):
    from src.orchestration import rag_pipeline as module
    monkeypatch.setenv('LLM_GENERATION_ENABLED', 'true')
    monkeypatch.setenv('ALLOW_GENERAL_KNOWLEDGE', 'true')
    monkeypatch.setattr(module, 'get_conversation_history', AsyncMock(return_value=([], None)))
    monkeypatch.setattr(module, 'get_lm_usage_since_split', lambda n: ({'num_calls':1}, {}))
    monkeypatch.setattr(module, 'log_step_timings', Mock())
    monkeypatch.setattr(module.dspy, 'settings', SimpleNamespace(lm=SimpleNamespace(history=[])))
    class Agent:
        def __init__(self, *a):pass
        async def stream_response(self, question):
            yield 'General answer'
    monkeypatch.setattr(module, 'GeneralKnowledgeAgent', Agent)
    rail_calls=[]
    async def rails(**kw):
        rail_calls.append(kw['user_message'])
        async for token in kw['bot_message_generator']:
            yield token
    service=SimpleNamespace(conversation_history_store=None,
        _refine_user_prompt=Mock(return_value=(SimpleNamespace(original_question='A question',refined_questions=['A question']),{})),
        _safe_retrieve_contextual_chunks=AsyncMock(return_value=chunks),
        format_sse=lambda chat,content:content, log_costs=Mock(),
        _extract_document_references=Mock(return_value=[]),
        langfuse_config=SimpleNamespace(langfuse_client=None),store_streaming_inference=AsyncMock())
    request=SimpleNamespace(chatId='fixture',message='A question',conversationHistory=[],environment='testing',connection_id='fixture')
    ctx=SimpleNamespace(stream_id='stream',token_count=0,mark_completed=Mock(),mark_cancelled=Mock(),mark_error=Mock())
    components={'llm_manager':Mock(),'contextual_retriever':Mock(),'response_generator':SimpleNamespace(check_scope_quick=AsyncMock(return_value=out_of_scope)), 'guardrails_adapter':SimpleNamespace(stream_with_guardrails=rails)}
    async def run():return [x async for x in module.RagPipeline(service).stream(request,components,ctx,{}, {})]
    result=asyncio.run(run())
    assert result==['General answer','END'] and rail_calls==['A question']
    service.store_streaming_inference.assert_awaited_once()
    service._extract_document_references.assert_called_once_with([])
    assert request._rag_ranked_chunks==[] and request._general_knowledge_answer is True


def test_nonstream_general_discards_unrelated_citations_and_counts_both_calls(monkeypatch):
    from contextlib import nullcontext
    from src.orchestration import rag_pipeline as module
    monkeypatch.setenv('LLM_GENERATION_ENABLED','true')
    monkeypatch.setenv('ALLOW_GENERAL_KNOWLEDGE','true')
    class Agent:
        def __init__(self,*a):pass
        def forward(self,**kw):return {'answer':'General answer','questionOutOfLLMScope':False,'usage':{'num_calls':1,'total_cost':.02}}
    monkeypatch.setattr(module,'GeneralKnowledgeAgent',Agent)
    normal=SimpleNamespace(forward=Mock(return_value={'answer':'Not enough context','questionOutOfLLMScope':True,'usage':{'num_calls':1,'total_cost':.01}}))
    source=SimpleNamespace(langfuse_config=SimpleNamespace(langfuse_client=None),_extract_document_references=Mock(return_value=[]),_format_chunks_for_test_response=Mock(return_value=[]))
    request=SimpleNamespace(chatId='fixture',environment='testing')
    costs={}
    result=module.RagPipeline(source).generate_response(SimpleNamespace(use_task_local=lambda:nullcontext()),request,SimpleNamespace(original_question='A question'),[{'text':'irrelevant'}],normal,costs)
    assert result.content=='General answer' and not result.questionOutOfLLMScope
    source._extract_document_references.assert_called_once_with([])
    assert request._general_knowledge_answer is True
    assert costs['response_generator']['num_calls']==2
    assert costs['response_generator']['total_cost']==pytest.approx(.03)


def test_source_answer_takes_precedence_over_general_knowledge(monkeypatch):
    from contextlib import nullcontext
    from src.orchestration import rag_pipeline as module
    monkeypatch.setenv('LLM_GENERATION_ENABLED','true')
    monkeypatch.setenv('ALLOW_GENERAL_KNOWLEDGE','true')
    never=Mock(side_effect=AssertionError('grounded answer must not call general model'))
    monkeypatch.setattr(module,'GeneralKnowledgeAgent',never)
    normal=SimpleNamespace(forward=Mock(return_value={'answer':'Official grounded answer','questionOutOfLLMScope':False,'usage':{}}))
    source=SimpleNamespace(langfuse_config=SimpleNamespace(langfuse_client=None),_extract_document_references=Mock(return_value=[]),_format_chunks_for_test_response=Mock(return_value=[]))
    request=SimpleNamespace(chatId='fixture',environment='testing');chunks=[{'text':'Relevant official source'}]
    result=module.RagPipeline(source).generate_response(SimpleNamespace(use_task_local=lambda:nullcontext()),request,SimpleNamespace(original_question='A question'),chunks,normal,{})
    assert result.content=='Official grounded answer'
    never.assert_not_called()
    source._extract_document_references.assert_called_once_with(chunks)
