"""Verified local FastEmbed model for isolated Central deployments."""
import hashlib
import json
import os
from pathlib import Path


def model_parameters():
    path = os.getenv("GUARDRAILS_EMBEDDING_PATH")
    if not path:
        return None
    return {"specific_model_path": path, "local_files_only": True, "threads": 2}


def verify_and_load():
    """Fail startup for missing/corrupt artifacts before any provider request."""
    parameters = model_parameters()
    if parameters is None:
        return False
    path = Path(parameters["specific_model_path"])
    manifest = json.loads((path / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError("invalid embedding artifact name")
        with (path / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError("guardrails embedding artifact checksum mismatch")
    from nemoguardrails.embeddings.providers.fastembed import FastEmbedEmbeddingModel

    model = FastEmbedEmbeddingModel("all-MiniLM-L6-v2", **parameters)
    if model.embedding_size != 384:
        raise ValueError("unexpected guardrails embedding dimension")
    return True


def configure(rails_config):
    parameters = model_parameters()
    if parameters is None:
        return
    from nemoguardrails.rails.llm.config import Model

    # Preserve the tuned rails and the same default embedding model/engine.
    rails_config.models = [m for m in rails_config.models if m.type != "embeddings"]
    rails_config.models.append(Model(type="embeddings", engine="FastEmbed",
                                     model="all-MiniLM-L6-v2", parameters=parameters))
