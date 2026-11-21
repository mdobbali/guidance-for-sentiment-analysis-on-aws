"""Text embeddings via Amazon Titan Text Embeddings V2 on Bedrock."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol


class Embedder(Protocol):
    model_id: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class TitanEmbedder:
    def __init__(self, bedrock_runtime, model_id: str, dimensions: int = 1024, workers: int = 8):
        self._client = bedrock_runtime
        self.model_id = model_id
        self._dimensions = dimensions
        self._workers = workers

    def _one(self, text: str) -> list[float]:
        resp = self._client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"inputText": text, "dimensions": self._dimensions, "normalize": True}),
        )
        return json.loads(resp["body"].read())["embedding"]

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Titan V2 takes one input per call; fan out a little for ingestion speed.
        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            return list(pool.map(self._one, texts))
