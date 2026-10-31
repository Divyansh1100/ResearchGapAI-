from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class ModelConfig:
    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    model_name: str = ""
    embedding_model: str = DEFAULT_EMBEDDING_MODEL

    @property
    def has_llm(self) -> bool:
        return bool(self.api_key) and bool(self.model_name)

    @property
    def has_embedding_model(self) -> bool:
        return bool(self.embedding_model)


def get_model_config() -> ModelConfig:
    return ModelConfig(
        api_key=os.getenv("LLM_API_KEY", "").strip(),
        base_url=(os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL) or DEFAULT_BASE_URL).strip(),
        model_name=os.getenv("LLM_MODEL", "").strip(),
        embedding_model=(os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL) or DEFAULT_EMBEDDING_MODEL).strip(),
    )
