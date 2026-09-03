from __future__ import annotations

import json
import random
from typing import Any

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError

PROBE_ROUND_SYSTEM_PROMPT = """You are a mechanistic interpretability probing agent.
You are given evidence about one sparse autoencoder feature.
Use only the evidence provided.
Your job is to propose a concrete feature hypothesis and a small set of tests.
Prefer minimal pairs, clean contrast cases, and controlled edits.
Do not produce more probes than requested.
Return exactly one JSON object with these keys:
- trigger_hypothesis: short text
- anti_trigger_hypothesis: short text
- confounds: list of short strings
- uncertainty: short text
- confidence: number from 0 to 1
- should_run_steering: boolean
- steering_reason: short text
- steering_candidate_text: string or null
- steering_positions: "all", "last", or a list of token indices
- synthetic_probes: list of objects with keys:
  - probe_id
  - text
  - expected_effect ("positive", "negative", "boundary", "confound", or "contrast")
  - probe_type
  - reason
- real_edits: list of objects with keys:
  - edit_id
  - source_id
  - edited_text
  - expected_effect ("positive", "negative", "boundary", "confound", or "contrast")
  - edit_type
  - reason
"""


PROBE_REPORT_SYSTEM_PROMPT = """You are summarizing the results of an interpretability probing run for one sparse autoencoder feature.
Use only the supplied evidence and test outcomes.
Return exactly one JSON object with these keys:
- final_hypothesis
- summary
- confidence
- uncertainty
- evidence_for
- evidence_against
- rejected_hypotheses
- remaining_open_questions
"""


PROBE_JSON_REPAIR_SYSTEM_PROMPT = """You repair malformed JSON produced by another model.
Return exactly one valid JSON object.
Do not add markdown fences or commentary.
Preserve the original meaning as closely as possible.
"""


class OpenAIProbingAgent:
    def __init__(self, api_key: str, model: str, timeout_seconds: float, max_retries: int) -> None:
        self.client = OpenAI(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def propose_round(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._create_response_with_backoff(PROBE_ROUND_SYSTEM_PROMPT, payload)
        result = self._parse_or_repair_json(response.output_text.strip())
        result["agent_model"] = self.model
        result["raw_response_text"] = response.output_text.strip()
        return result

    def synthesize_report(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._create_response_with_backoff(PROBE_REPORT_SYSTEM_PROMPT, payload)
        result = self._parse_or_repair_json(response.output_text.strip())
        result["agent_model"] = self.model
        result["raw_response_text"] = response.output_text.strip()
        return result

    def _create_response_with_backoff(self, system_prompt: str, payload: dict[str, Any]):
        attempts = max(1, self.max_retries + 1)
        for attempt in range(attempts):
            try:
                return self.client.responses.create(
                    model=self.model,
                    input=[
                        {
                            "role": "system",
                            "content": [{"type": "input_text", "text": system_prompt}],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                                }
                            ],
                        },
                    ],
                )
            except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError):
                if attempt >= attempts - 1:
                    raise
                delay = min(90.0, 4.0 * (2**attempt) + random.uniform(0.0, 1.5))
                import time

                time.sleep(delay)

    def _parse_or_repair_json(self, text: str) -> dict[str, Any]:
        try:
            return self._extract_json(text)
        except (json.JSONDecodeError, ValueError):
            repaired = self._repair_json(text)
            result = self._extract_json(repaired)
            result["raw_repaired_response_text"] = repaired.strip()
            return result

    def _repair_json(self, broken_text: str) -> str:
        repair_payload = {"malformed_json": broken_text}
        response = self._create_response_with_backoff(PROBE_JSON_REPAIR_SYSTEM_PROMPT, repair_payload)
        return response.output_text.strip()

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if len(lines) >= 3:
                stripped = "\n".join(lines[1:-1]).strip()
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("Probing agent response did not contain a JSON object.")
        candidate = stripped[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            decoder = json.JSONDecoder()
            for offset in range(start, len(stripped)):
                if stripped[offset] != "{":
                    continue
                try:
                    parsed, end_index = decoder.raw_decode(stripped[offset:])
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    return parsed
            raise


