"""OpenAI-backed judge clients that label features from collected evidence."""

from __future__ import annotations

import asyncio
import json
import random
import time
from typing import Any

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, InternalServerError, OpenAI, RateLimitError

JUDGE_SYSTEM_PROMPT = """You are an interpretability researcher analyzing sparse autoencoder features.
Use only the evidence provided.
Do not invent evidence that is not present.
Return a single JSON object with these keys:
- conceptual_label: short name for the feature
- feature_type: one of lexical, subword, entity, topical, syntactic, discourse, semantic, formatting, other
- summary: 1-3 sentence explanation of what the feature appears to represent
- transcript_relevance_rationale: why the feature seems relevant to the transcript stimuli
- evidence_for: list of short evidence-backed points
- evidence_against: list of short uncertainty or counterevidence points
- confidence: number from 0 to 1
- uncertainty: short note describing ambiguity or missing evidence
- follow_up: short recommendation for next analysis step
"""



def extract_json_object(text: str, label: str = "Model") -> dict[str, Any]:
    """Return the first JSON object in a model response, tolerating code fences and surrounding prose."""

    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"{label} response did not contain a JSON object.")
    candidate = stripped[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for offset in range(start, len(stripped)):
            if stripped[offset] != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(stripped[offset:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        raise


class OpenAIJudge:
    def __init__(self, api_key: str, model: str, timeout_seconds: float = 60.0, max_retries: int = 2) -> None:
        self.client = OpenAI(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def judge_feature(self, evidence: dict[str, Any]) -> dict[str, Any]:
        response = self._create_response_with_backoff(evidence)
        output_text = response.output_text.strip()
        payload = extract_json_object(output_text, label="Judge")
        payload["judge_model"] = self.model
        payload["raw_response_text"] = output_text
        return payload

    def _prompt_from_evidence(self, evidence: dict[str, Any]) -> str:
        return json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))

    def _create_response_with_backoff(self, evidence: dict[str, Any]):
        attempts = max(1, self.max_retries + 1)
        for attempt in range(attempts):
            try:
                return self.client.responses.create(
                    model=self.model,
                    input=[
                        {
                            "role": "system",
                            "content": [{"type": "input_text", "text": JUDGE_SYSTEM_PROMPT}],
                        },
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": self._prompt_from_evidence(evidence)}],
                        },
                    ],
                )
            except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError):
                if attempt >= attempts - 1:
                    raise
                self._sleep_backoff(attempt)

    @staticmethod
    def _sleep_backoff(attempt: int) -> None:
        delay = min(90.0, 4.0 * (2**attempt) + random.uniform(0.0, 1.5))

        time.sleep(delay)



class AsyncOpenAIJudge:
    def __init__(self, api_key: str, model: str, timeout_seconds: float = 60.0, max_retries: int = 2) -> None:
        self.client = AsyncOpenAI(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    async def judge_feature(self, evidence: dict[str, Any]) -> dict[str, Any]:
        response = await self._create_response_with_backoff(evidence)
        output_text = response.output_text.strip()
        payload = extract_json_object(output_text, label="Judge")
        payload["judge_model"] = self.model
        payload["raw_response_text"] = output_text
        return payload

    async def _create_response_with_backoff(self, evidence: dict[str, Any]):
        attempts = max(1, self.max_retries + 1)
        for attempt in range(attempts):
            try:
                return await self.client.responses.create(
                    model=self.model,
                    input=[
                        {
                            "role": "system",
                            "content": [{"type": "input_text", "text": JUDGE_SYSTEM_PROMPT}],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": json.dumps(evidence, ensure_ascii=False, separators=(",", ":")),
                                }
                            ],
                        },
                    ],
                )
            except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError):
                if attempt >= attempts - 1:
                    raise
                delay = min(90.0, 4.0 * (2**attempt) + random.uniform(0.0, 1.5))
                await asyncio.sleep(delay)

