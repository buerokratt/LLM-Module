import json
import os
from pathlib import Path

import pytest

from src.guardrails.offline_embeddings import configure, verify_and_load


def test_no_local_override_preserves_upstream_config(monkeypatch):
    monkeypatch.delenv("GUARDRAILS_EMBEDDING_PATH", raising=False)
    from nemoguardrails import RailsConfig
    config = RailsConfig.from_content(yaml_content="models: []")
    configure(config)
    assert config.models == []
    assert verify_and_load() is False


def test_missing_and_corrupt_models_fail_before_loading(tmp_path, monkeypatch):
    monkeypatch.setenv("GUARDRAILS_EMBEDDING_PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        verify_and_load()
    (tmp_path / "model.onnx").write_bytes(b"corrupt")
    (tmp_path / "manifest.json").write_text(json.dumps({"files": {"model.onnx": "0" * 64}}))
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_and_load()


@pytest.mark.skipif(not os.getenv("GUARDRAILS_EMBEDDING_PATH"), reason="image artifact required")
def test_real_nemo_embeddings_without_downloads(monkeypatch):
    # Neither HF nor Google fallback may be used, even on a host with internet.
    from fastembed.common.model_management import ModelManagement
    def forbidden(*args, **kwargs):
        raise AssertionError("runtime model download attempted")
    monkeypatch.setattr(ModelManagement, "download_files_from_huggingface", forbidden)
    monkeypatch.setattr(ModelManagement, "retrieve_model_gcs", forbidden)
    assert verify_and_load() is True
    from nemoguardrails import LLMRails, RailsConfig
    from langchain_core.language_models.fake import FakeListLLM
    config = RailsConfig.from_content(
        yaml_content="models: []",
        colang_content='define user express greeting\n  "hello"\n  "hi"\ndefine bot express greeting\n  "Hello!"\ndefine flow greeting\n  user express greeting\n  bot express greeting\n',
    )
    configure(config)
    rails = LLMRails(config, llm=FakeListLLM(responses=["Hello!"]))
    assert rails.default_embedding_model == "all-MiniLM-L6-v2"
