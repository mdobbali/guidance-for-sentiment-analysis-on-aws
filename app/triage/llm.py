"""Claude on Amazon Bedrock, called through the Anthropic SDK's Bedrock Mantle client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Generation:
    text: str | None
    stop_reason: str | None
    model: str
    input_tokens: int
    output_tokens: int
    refusal_category: str | None = None


class Generator(Protocol):
    model_id: str

    def generate(self, system: str, user: str, schema: dict) -> Generation: ...


class ClaudeBedrockGenerator:
    """Structured-output generation. The JSON schema is enforced by the API
    (output_config.format), so the response always parses."""

    def __init__(self, region: str, model_id: str, effort: str = "medium", fallback_model: str | None = None):
        from anthropic import AnthropicBedrockMantle

        middleware = []
        if fallback_model:
            # Retries a refused request once on the fallback model. Pharma and
            # clinical text can trip safety classifiers on legitimate questions.
            from anthropic.lib.middleware import BetaRefusalFallbackMiddleware

            middleware.append(BetaRefusalFallbackMiddleware([{"model": fallback_model}]))
        self._client = AnthropicBedrockMantle(aws_region=region, middleware=middleware)
        self.model_id = model_id
        self._effort = effort

    def generate(self, system: str, user: str, schema: dict) -> Generation:
        resp = self._client.beta.messages.create(
            model=self.model_id,
            max_tokens=8000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self._effort, "format": {"type": "json_schema", "schema": schema}},
        )
        text = next((b.text for b in resp.content if b.type == "text"), None)
        category = None
        if resp.stop_reason == "refusal" and getattr(resp, "stop_details", None):
            category = getattr(resp.stop_details, "category", None)
        return Generation(
            text=text,
            stop_reason=resp.stop_reason,
            model=resp.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            refusal_category=category,
        )
