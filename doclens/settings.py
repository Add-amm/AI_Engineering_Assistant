"""Runtime options. The default configuration works without network access."""

import os
from dataclasses import dataclass
from typing import Literal, cast

Provider = Literal["extractive", "openai", "ollama"]
Embeddings = Literal["lsa", "sentence-transformers"]


@dataclass(frozen=True)
class Settings:
    embeddings: Embeddings = "lsa"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_revision: str | None = None
    rerank_model: str | None = None
    rerank_revision: str | None = None
    local_models_only: bool = False
    provider: Provider = "extractive"
    model: str | None = None
    llm_base_url: str | None = None
    llm_api_key: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        defaults = cls()
        provider = os.getenv("DOCLENS_PROVIDER", defaults.provider).strip().lower()
        embeddings = os.getenv("DOCLENS_EMBEDDINGS", defaults.embeddings).strip().lower()
        if provider not in {"extractive", "openai", "ollama"}:
            raise ValueError("DOCLENS_PROVIDER must be extractive, openai or ollama.")
        if embeddings not in {"lsa", "sentence-transformers"}:
            raise ValueError("DOCLENS_EMBEDDINGS must be lsa or sentence-transformers.")
        embedding_model = os.getenv("DOCLENS_EMBEDDING_MODEL") or defaults.embedding_model
        # Lock the upstream snapshot used by the documented baseline.
        default_revision = (
            "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
            if embedding_model == defaults.embedding_model else None
        )
        return cls(
            embeddings=cast(Embeddings, embeddings),
            embedding_model=embedding_model,
            embedding_revision=os.getenv("DOCLENS_EMBEDDING_REVISION") or default_revision,
            rerank_model=os.getenv("DOCLENS_RERANK_MODEL"),
            rerank_revision=os.getenv("DOCLENS_RERANK_REVISION"),
            local_models_only=os.getenv("DOCLENS_LOCAL_MODELS_ONLY", "").lower() in {"1", "true"},
            provider=cast(Provider, provider),
            model=os.getenv("DOCLENS_MODEL"),
            llm_base_url=os.getenv("DOCLENS_LLM_BASE_URL"),
            llm_api_key=os.getenv("DOCLENS_LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
        )
