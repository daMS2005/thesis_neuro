from __future__ import annotations

import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any, ClassVar

import torch

from thesis_neuro.audit_data import (
    build_feature_lookup,
    build_script_audit_view,
    load_audit_bundle,
    resolve_audit_paths,
)
from thesis_neuro.config import AppConfig
from thesis_neuro.models import GemmaModelAdapter
from thesis_neuro.probes.judge import OpenAIProbingAgent
from thesis_neuro.probes.schema import ProbeRunPaths
from thesis_neuro.sae import GemmaScopeAdapter


class FeatureProbingPipeline:
    _RUNTIME_CACHE: ClassVar[
        dict[tuple[str, str, str, str, str], tuple[GemmaModelAdapter, GemmaScopeAdapter]]
    ] = {}

    def __init__(
        self,
        config: AppConfig,
        analysis_dir: str | None,
        transcript_dir: str | None,
        dolma_dir: str | None,
        alignment_path: str | None,
        layer: int,
        feature_id: int,
        script_id: str | None = None,
        max_rounds: int | None = None,
        run_steering: bool = False,
        judge_model: str | None = None,
    ) -> None:
        self.config = config
        self.analysis_dir = Path(analysis_dir or config.analysis.transcript_output_dir or config.output.dir)
        self.transcript_dir = Path(transcript_dir or config.analysis.transcript_output_dir or self.analysis_dir)
        self.dolma_dir = Path(dolma_dir or config.analysis.dolma_output_dir or self.analysis_dir)
        self.alignment_path = Path(
            alignment_path
            or config.analysis.alignment_path
            or (self.analysis_dir / "feature_alignment.jsonl")
        )
        self.layer = int(layer)
        self.feature_id = int(feature_id)
        self.script_id = script_id
        self.max_rounds = int(max_rounds or config.probing.max_rounds)
        self.run_steering = bool(run_steering or config.probing.enable_steering)
        self.agent_model = str(judge_model or config.probing.model or config.judge.model)
        self.bundle_id = self._slugify(self.analysis_dir.name or config.model.base_model_id)
        self.paths = self._build_paths()
        self.bundle = None
        self.model: GemmaModelAdapter | None = None
        self.sae: GemmaScopeAdapter | None = None

    def run(self) -> dict[str, Any]:
        if not self.config.env.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for feature probing.")

        self.paths.root.mkdir(parents=True, exist_ok=True)
        evidence = self._load_or_build_evidence()
        prior_rounds = self._load_jsonl(self.paths.rounds_path)
        prior_tests = self._load_jsonl(self.paths.tests_path)
        prior_steering = self._load_jsonl(self.paths.steering_path)

        agent = OpenAIProbingAgent(
            api_key=self.config.env.openai_api_key,
            model=self.agent_model,
            timeout_seconds=self.config.judge.timeout_seconds,
            max_retries=self.config.judge.max_retries,
        )

        round_index = len(prior_rounds) + 1
        while round_index <= self.max_rounds:
            previous_summary = prior_rounds[-1]["round_summary"] if prior_rounds else None
            if previous_summary is not None:
                if float(previous_summary.get("agent_confidence", 0.0)) >= float(self.config.probing.stop_confidence):
                    break
                if self._stopping_no_gain(prior_rounds):
                    break

            round_plan = agent.propose_round(
                self._round_prompt_payload(
                    evidence=evidence,
                    prior_rounds=prior_rounds,
                    prior_tests=prior_tests,
                    prior_steering=prior_steering,
                    round_index=round_index,
                )
            )
            synthetic_probes = self._normalize_synthetic_probes(round_plan.get("synthetic_probes", []))
            real_edits = self._normalize_real_edits(round_plan.get("real_edits", []))
            synthetic_probes = synthetic_probes[: self.config.probing.synthetic_probes_per_round]
            real_edits = real_edits[: self.config.probing.real_edits_per_round]

            round_test_rows = self._evaluate_round_tests(
                evidence=evidence,
                round_index=round_index,
                synthetic_probes=synthetic_probes,
                real_edits=real_edits,
            )
            for row in round_test_rows:
                self._append_jsonl(self.paths.tests_path, row)
            prior_tests.extend(round_test_rows)

            steering_rows: list[dict[str, Any]] = []
            if self.run_steering and bool(round_plan.get("should_run_steering", False)):
                steering_rows = self._run_round_steering(
                    round_index=round_index,
                    round_plan=round_plan,
                    test_rows=round_test_rows,
                )
                for row in steering_rows:
                    self._append_jsonl(self.paths.steering_path, row)
                prior_steering.extend(steering_rows)

            round_record = {
                "round": round_index,
                "layer": self.layer,
                "feature_id": self.feature_id,
                "round_plan": round_plan,
                "round_summary": self._summarize_round(
                    round_plan=round_plan,
                    test_rows=round_test_rows,
                    steering_rows=steering_rows,
                ),
            }
            self._append_jsonl(self.paths.rounds_path, round_record)
            prior_rounds.append(round_record)
            round_index += 1

        report = self._synthesize_report(
            agent=agent,
            evidence=evidence,
            rounds=prior_rounds,
            tests=prior_tests,
            steering_rows=prior_steering,
        )
        self.paths.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        manifest = {
            "bundle_id": self.bundle_id,
            "analysis_dir": str(self.analysis_dir),
            "transcript_dir": str(self.transcript_dir),
            "dolma_dir": str(self.dolma_dir),
            "alignment_path": str(self.alignment_path) if self.alignment_path.exists() else None,
            "layer": self.layer,
            "feature_id": self.feature_id,
            "script_id": self.script_id,
            "agent_model": self.agent_model,
            "max_rounds": self.max_rounds,
            "run_steering": self.run_steering,
            "rounds_written": len(prior_rounds),
            "tests_written": len(prior_tests),
            "steering_rows_written": len(prior_steering),
            "artifacts": {
                "evidence": str(self.paths.evidence_path),
                "rounds": str(self.paths.rounds_path),
                "tests": str(self.paths.tests_path),
                "steering": str(self.paths.steering_path),
                "report": str(self.paths.report_path),
            },
        }
        self.paths.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest

    def _load_or_build_evidence(self) -> dict[str, Any]:
        self.paths.root.mkdir(parents=True, exist_ok=True)
        if self.paths.evidence_path.exists():
            cached = json.loads(self.paths.evidence_path.read_text(encoding="utf-8"))
            if self._evidence_is_usable(cached):
                return cached

        paths = resolve_audit_paths(
            analysis_dir=self.analysis_dir,
            transcript_dir=self.transcript_dir,
            dolma_dir=self.dolma_dir,
        )
        bundle = load_audit_bundle(paths)
        self.bundle = bundle
        selected_script = (
            next((item for item in bundle.scripts if str(item.get("script_id")) == self.script_id), None)
            if self.script_id
            else None
        )

        detail = {
            "layer": self.layer,
            "feature_id": self.feature_id,
        }
        if selected_script is not None:
            script_id = str(selected_script["script_id"])
            script_view = build_script_audit_view(bundle, script_id=script_id, layer=self.layer)
            detail = build_feature_lookup(bundle, self.layer, script_view, self.feature_id)
            detail["script_id"] = script_id
            detail["script_meta"] = selected_script
        else:
            detail = self._fallback_feature_detail(bundle)

        source_texts = self._source_texts(detail)
        evidence = {
            "bundle_id": self.bundle_id,
            "model_id": self.config.model.base_model_id,
            "scope_release": self.config.model.scope_release,
            "layer": self.layer,
            "feature_id": self.feature_id,
            "script_id": detail.get("script_id"),
            "feature_evidence": detail,
            "source_texts": source_texts,
        }
        self.paths.evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        return evidence

    def _evidence_is_usable(self, evidence: dict[str, Any]) -> bool:
        if not isinstance(evidence, dict):
            return False
        if int(evidence.get("layer", -1)) != self.layer or int(evidence.get("feature_id", -1)) != self.feature_id:
            return False
        feature_evidence = evidence.get("feature_evidence")
        if not isinstance(feature_evidence, dict):
            return False
        if not evidence.get("bundle_id"):
            return False
        if not evidence.get("source_texts"):
            return False
        return True

    def _fallback_feature_detail(self, bundle: Any) -> dict[str, Any]:
        key = (self.layer, self.feature_id)
        relevance = bundle.relevance_by_key.get(key, {})
        concept = bundle.concepts_by_key.get(key, {})
        return {
            "layer": self.layer,
            "feature_id": self.feature_id,
            "label": (concept.get("judge_output") or {}).get("conceptual_label"),
            "feature_type": (concept.get("judge_output") or {}).get("feature_type"),
            "summary": (concept.get("judge_output") or {}).get("summary"),
            "transcript_rationale": (concept.get("judge_output") or {}).get("transcript_relevance_rationale"),
            "transcript_relevance_rank": relevance.get("transcript_relevance_rank"),
            "transcript_relevance_score": relevance.get("transcript_relevance_score"),
            "top_transcript_examples": relevance.get("top_transcript_examples", []),
            "top_dolma_contexts": relevance.get("top_dolma_contexts", []),
            "top_correlated_features": bundle.correlations_by_key.get(key, {}).get("top_correlated_features", []),
            "alignment_summary": bundle.alignments_by_key.get(key, [])[:2],
            "judge_label": (concept.get("judge_output") or {}).get("conceptual_label"),
            "judge_summary": (concept.get("judge_output") or {}).get("summary"),
            "judge_evidence_for": (concept.get("judge_output") or {}).get("evidence_for", []),
            "judge_evidence_against": (concept.get("judge_output") or {}).get("evidence_against", []),
            "judge_uncertainty": (concept.get("judge_output") or {}).get("uncertainty"),
            "judge_follow_up": (concept.get("judge_output") or {}).get("follow_up", []),
            "judge_coverage_status": concept.get("judge_status", "missing"),
        }

    def _source_texts(self, detail: dict[str, Any]) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        for index, example in enumerate(detail.get("top_transcript_examples", []), start=1):
            text = self._normalize_source_text(example.get("text")) or " ".join(example.get("snippet_tokens") or [])
            if not text:
                continue
            sources.append(
                {
                    "source_id": f"transcript_{index}",
                    "source_type": "transcript_example",
                    "text": text,
                    "selection_reason": example.get("selection_reason"),
                    "activation": float(example.get("activation", 0.0)),
                }
            )
        for index, context in enumerate(detail.get("top_dolma_contexts", []), start=1):
            text = self._dolma_context_text(context)
            if not text:
                continue
            sources.append(
                {
                    "source_id": f"dolma_{index}",
                    "source_type": "dolma_context",
                    "text": text,
                    "selection_reason": context.get("selection_reason"),
                    "dominant_scale": context.get("dominant_scale"),
                    "feature_activation_total": float(context.get("feature_activation_total", 0.0)),
                    "feature_activation_peak": float(context.get("feature_activation_peak", 0.0)),
                }
            )
        for index, item in enumerate(detail.get("script_sentence_hits", []), start=1):
            text = self._normalize_source_text(item.get("sentence_text"))
            if not text:
                continue
            sources.append(
                {
                    "source_id": f"script_sentence_{index}",
                    "source_type": "script_sentence",
                    "text": text,
                    "total_activation": float(item.get("total_activation", 0.0)),
                    "max_activation": float(item.get("max_activation", 0.0)),
                }
            )
        return sources[:12]

    def _round_prompt_payload(
        self,
        evidence: dict[str, Any],
        prior_rounds: list[dict[str, Any]],
        prior_tests: list[dict[str, Any]],
        prior_steering: list[dict[str, Any]],
        round_index: int,
    ) -> dict[str, Any]:
        compact_rounds = [
            {
                "round": row.get("round"),
                "trigger_hypothesis": (row.get("round_plan") or {}).get("trigger_hypothesis"),
                "anti_trigger_hypothesis": (row.get("round_plan") or {}).get("anti_trigger_hypothesis"),
                "confidence": (row.get("round_summary") or {}).get("agent_confidence"),
                "support_score": (row.get("round_summary") or {}).get("support_score"),
            }
            for row in prior_rounds[-2:]
        ]
        compact_tests = [
            {
                "round": row.get("round"),
                "test_kind": row.get("test_kind"),
                "probe_id": row.get("probe_id"),
                "expected_effect": row.get("expected_effect"),
                "feature_total_activation": row.get("feature_total_activation"),
                "comparison_delta": row.get("comparison_delta"),
            }
            for row in prior_tests[-12:]
        ]
        compact_steering = [
            {
                "round": row.get("round"),
                "steering_strength": row.get("steering_strength"),
                "target_total_delta": row.get("target_total_delta"),
                "target_peak_delta": row.get("target_peak_delta"),
            }
            for row in prior_steering[-6:]
        ]
        detail = evidence["feature_evidence"]
        return {
            "round": round_index,
            "feature": {
                "bundle_id": evidence["bundle_id"],
                "model_id": evidence["model_id"],
                "scope_release": evidence["scope_release"],
                "layer": evidence["layer"],
                "feature_id": evidence["feature_id"],
                "label": detail.get("judge_label") or detail.get("label"),
                "summary": detail.get("judge_summary") or detail.get("summary"),
                "transcript_rationale": detail.get("transcript_rationale"),
                "judge_evidence_for": self._ensure_list(detail.get("judge_evidence_for"))[:5],
                "judge_evidence_against": self._ensure_list(detail.get("judge_evidence_against"))[:5],
                "judge_uncertainty": detail.get("judge_uncertainty"),
                "top_correlated_features": detail.get("top_correlated_features", [])[:5],
                "alignment_summary": detail.get("alignment_summary", [])[:2],
            },
            "source_texts": evidence.get("source_texts", [])[:8],
            "prior_rounds": compact_rounds,
            "recent_tests": compact_tests,
            "recent_steering": compact_steering,
            "constraints": {
                "synthetic_probe_limit": self.config.probing.synthetic_probes_per_round,
                "real_edit_limit": self.config.probing.real_edits_per_round,
                "enable_steering": self.run_steering,
            },
        }

    def _evaluate_round_tests(
        self,
        evidence: dict[str, Any],
        round_index: int,
        synthetic_probes: list[dict[str, Any]],
        real_edits: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        source_lookup = {item["source_id"]: item for item in evidence.get("source_texts", [])}

        for probe in synthetic_probes:
            metrics = self.score_feature_on_texts([probe.get("text", "")])[0]
            rows.append(
                {
                    "round": round_index,
                    "test_kind": "synthetic_probe",
                    "probe_id": probe.get("probe_id"),
                    "expected_effect": probe.get("expected_effect"),
                    "probe_type": probe.get("probe_type"),
                    "reason": probe.get("reason"),
                    "text": probe.get("text"),
                    **metrics,
                    "nearest_contexts": self.nearest_contexts_for_text(
                        probe.get("text", ""),
                        evidence.get("source_texts", []),
                    ),
                }
            )

        for edit in real_edits:
            source = source_lookup.get(str(edit.get("source_id", "")))
            if source is None:
                continue
            original_metrics, edited_metrics = self.score_feature_with_counterfactuals(
                original_text=str(source.get("text", "")),
                edited_text=str(edit.get("edited_text", "")),
            )
            rows.append(
                {
                    "round": round_index,
                    "test_kind": "real_edit",
                    "probe_id": edit.get("edit_id"),
                    "source_id": edit.get("source_id"),
                    "expected_effect": edit.get("expected_effect"),
                    "edit_type": edit.get("edit_type"),
                    "reason": edit.get("reason"),
                    "source_text": source.get("text"),
                    "edited_text": edit.get("edited_text"),
                    "original_metrics": original_metrics,
                    "edited_metrics": edited_metrics,
                    "feature_total_activation": edited_metrics["feature_total_activation"],
                    "feature_peak_activation": edited_metrics["feature_peak_activation"],
                    "active_token_fraction": edited_metrics["active_token_fraction"],
                    "comparison_delta": float(
                        edited_metrics["feature_total_activation"] - original_metrics["feature_total_activation"]
                    ),
                    "nearest_contexts": self.nearest_contexts_for_text(
                        str(edit.get("edited_text", "")),
                        evidence.get("source_texts", []),
                    ),
                }
            )
        return rows

    def _run_round_steering(
        self,
        round_index: int,
        round_plan: dict[str, Any],
        test_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        candidate_text = self._normalize_source_text(round_plan.get("steering_candidate_text"))
        if not candidate_text:
            positive_rows = [
                row
                for row in test_rows
                if row.get("test_kind") == "synthetic_probe" and row.get("expected_effect") == "positive"
            ]
            positive_rows = sorted(
                positive_rows,
                key=lambda row: float(row.get("feature_total_activation", 0.0)),
                reverse=True,
            )
            candidate_text = positive_rows[0]["text"] if positive_rows else ""
        if not candidate_text:
            return []

        positions = round_plan.get("steering_positions", "all")
        rows: list[dict[str, Any]] = []
        for strength in self.config.probing.steering_strengths:
            rows.append(
                self.steer_feature_on_text(
                    text=candidate_text,
                    round_index=round_index,
                    steering_strength=float(strength),
                    steering_positions=positions,
                    steering_reason=round_plan.get("steering_reason"),
                )
            )
        return rows

    def _summarize_round(
        self,
        round_plan: dict[str, Any],
        test_rows: list[dict[str, Any]],
        steering_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        positive_scores = [
            float(row.get("feature_total_activation", 0.0))
            for row in test_rows
            if row.get("expected_effect") == "positive"
        ]
        negative_scores = [
            float(row.get("feature_total_activation", 0.0))
            for row in test_rows
            if row.get("expected_effect") == "negative"
        ]
        edit_deltas = [
            float(row.get("comparison_delta", 0.0))
            for row in test_rows
            if row.get("test_kind") == "real_edit"
        ]
        steering_best = max(
            (float(row.get("target_total_delta", 0.0)) for row in steering_rows),
            default=0.0,
        )
        support_score = (
            (mean(positive_scores) if positive_scores else 0.0)
            - (mean(negative_scores) if negative_scores else 0.0)
            + (mean(edit_deltas) if edit_deltas else 0.0)
            + steering_best
        )
        return {
            "trigger_hypothesis": round_plan.get("trigger_hypothesis"),
            "anti_trigger_hypothesis": round_plan.get("anti_trigger_hypothesis"),
            "agent_confidence": float(round_plan.get("confidence", 0.0) or 0.0),
            "uncertainty": round_plan.get("uncertainty"),
            "support_score": float(support_score),
            "positive_probe_mean": float(mean(positive_scores)) if positive_scores else 0.0,
            "negative_probe_mean": float(mean(negative_scores)) if negative_scores else 0.0,
            "edit_delta_mean": float(mean(edit_deltas)) if edit_deltas else 0.0,
            "best_steering_total_delta": float(steering_best),
            "steering_ran": bool(steering_rows),
        }

    def _synthesize_report(
        self,
        agent: OpenAIProbingAgent,
        evidence: dict[str, Any],
        rounds: list[dict[str, Any]],
        tests: list[dict[str, Any]],
        steering_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        report_input = {
            "feature": {
                "bundle_id": evidence["bundle_id"],
                "model_id": evidence["model_id"],
                "scope_release": evidence["scope_release"],
                "layer": evidence["layer"],
                "feature_id": evidence["feature_id"],
            },
            "evidence": {
                "feature_label": (evidence["feature_evidence"].get("judge_label") or evidence["feature_evidence"].get("label")),
                "feature_summary": (evidence["feature_evidence"].get("judge_summary") or evidence["feature_evidence"].get("summary")),
                "judge_evidence_for": self._ensure_list(evidence["feature_evidence"].get("judge_evidence_for"))[:5],
                "judge_evidence_against": self._ensure_list(evidence["feature_evidence"].get("judge_evidence_against"))[:5],
                "source_texts": evidence.get("source_texts", [])[:8],
            },
            "rounds": [
                {
                    "round": row.get("round"),
                    "plan": {
                        "trigger_hypothesis": (row.get("round_plan") or {}).get("trigger_hypothesis"),
                        "anti_trigger_hypothesis": (row.get("round_plan") or {}).get("anti_trigger_hypothesis"),
                        "confounds": self._ensure_list((row.get("round_plan") or {}).get("confounds")),
                        "uncertainty": (row.get("round_plan") or {}).get("uncertainty"),
                        "confidence": (row.get("round_plan") or {}).get("confidence"),
                    },
                    "summary": row.get("round_summary"),
                }
                for row in rounds
            ],
            "strongest_positive_probes": self._top_probe_rows(tests, expected="positive"),
            "strongest_negative_probes": self._top_probe_rows(tests, expected="negative"),
            "strongest_counterfactual_edits": self._top_edit_rows(tests),
            "steering_summary": self._compact_steering_rows(steering_rows),
        }
        try:
            agent_report = agent.synthesize_report(report_input)
            report_generation_error = None
        except Exception as exc:
            agent_report = self._fallback_report_payload(
                evidence=evidence,
                rounds=rounds,
                tests=tests,
                steering_rows=steering_rows,
            )
            report_generation_error = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        return {
            "bundle_id": evidence["bundle_id"],
            "layer": evidence["layer"],
            "feature_id": evidence["feature_id"],
            "feature_label": evidence["feature_evidence"].get("judge_label") or evidence["feature_evidence"].get("label"),
            "source_evidence_summary": {
                "top_transcript_examples": evidence["feature_evidence"].get("top_transcript_examples", [])[:5],
                "top_dolma_contexts": evidence["feature_evidence"].get("top_dolma_contexts", [])[:5],
                "judge_summary": evidence["feature_evidence"].get("judge_summary") or evidence["feature_evidence"].get("summary"),
            },
            "final_hypothesis": agent_report.get("final_hypothesis"),
            "summary": agent_report.get("summary"),
            "confidence": agent_report.get("confidence"),
            "uncertainty": agent_report.get("uncertainty"),
            "evidence_for": self._ensure_list(agent_report.get("evidence_for")),
            "evidence_against": self._ensure_list(agent_report.get("evidence_against")),
            "rejected_hypotheses": self._ensure_list(agent_report.get("rejected_hypotheses")),
            "remaining_open_questions": self._ensure_list(agent_report.get("remaining_open_questions")),
            "round_count": len(rounds),
            "strongest_positive_probes": self._top_probe_rows(tests, expected="positive"),
            "strongest_negative_probes": self._top_probe_rows(tests, expected="negative"),
            "strongest_counterfactual_edits": self._top_edit_rows(tests),
            "steering_summary": self._compact_steering_rows(steering_rows),
            "raw_agent_report": agent_report,
            "report_generation_error": report_generation_error,
        }

    def _fallback_report_payload(
        self,
        evidence: dict[str, Any],
        rounds: list[dict[str, Any]],
        tests: list[dict[str, Any]],
        steering_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        latest_round = rounds[-1] if rounds else {}
        latest_plan = latest_round.get("round_plan") or {}
        latest_summary = latest_round.get("round_summary") or {}
        positive = self._top_probe_rows(tests, expected="positive")
        negative = self._top_probe_rows(tests, expected="negative")
        edits = self._top_edit_rows(tests)
        steering = self._compact_steering_rows(steering_rows)
        support_bits: list[str] = []
        if positive:
            support_bits.append(f"{len(positive)} positive probes retained")
        if negative:
            support_bits.append(f"{len(negative)} negative probes retained")
        if edits:
            support_bits.append(f"{len(edits)} counterfactual edits scored")
        if steering:
            support_bits.append(f"{len(steering)} steering checks ran")
        summary = "; ".join(support_bits) if support_bits else "Probe evidence collected, but the final agent summary was unavailable."
        return {
            "final_hypothesis": latest_plan.get("trigger_hypothesis")
            or evidence["feature_evidence"].get("judge_label")
            or evidence["feature_evidence"].get("label"),
            "summary": summary,
            "confidence": latest_summary.get("agent_confidence", 0.0),
            "uncertainty": latest_plan.get("uncertainty")
            or evidence["feature_evidence"].get("judge_uncertainty")
            or "Final synthesized report fell back to deterministic summary after agent output parsing failed.",
            "evidence_for": [
                row.get("reason") or row.get("text") or row.get("edited_text")
                for row in positive[:3] + edits[:2]
                if row.get("reason") or row.get("text") or row.get("edited_text")
            ],
            "evidence_against": [
                row.get("reason") or row.get("text")
                for row in negative[:3]
                if row.get("reason") or row.get("text")
            ],
            "rejected_hypotheses": [],
            "remaining_open_questions": [
                "Need a clean agent-synthesized report to confirm the final natural-language interpretation."
            ],
            "fallback_generated": True,
        }

    def score_feature_on_texts(self, texts: list[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for start in range(0, len(texts), self.config.probing.max_batch_size):
            for text in texts[start : start + self.config.probing.max_batch_size]:
                rows.append(self._score_single_text(text))
        return rows

    def score_feature_with_counterfactuals(self, original_text: str, edited_text: str) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._score_single_text(original_text), self._score_single_text(edited_text)

    def nearest_contexts_for_text(self, text: str, source_texts: list[dict[str, Any]], top_n: int = 3) -> list[dict[str, Any]]:
        probe_vector = self._bag_of_words(text)
        if not probe_vector:
            return []
        scored: list[dict[str, Any]] = []
        for source in source_texts:
            source_text = str(source.get("text", ""))
            similarity = self._cosine_similarity(probe_vector, self._bag_of_words(source_text))
            if similarity <= 0:
                continue
            scored.append(
                {
                    "source_id": source.get("source_id"),
                    "source_type": source.get("source_type"),
                    "similarity": float(similarity),
                    "text": source_text,
                }
            )
        return sorted(scored, key=lambda row: row["similarity"], reverse=True)[:top_n]

    def steer_feature_on_text(
        self,
        text: str,
        round_index: int,
        steering_strength: float,
        steering_positions: list[int] | str | None,
        steering_reason: str | None = None,
    ) -> dict[str, Any]:
        self._ensure_runtime()
        assert self.model is not None
        assert self.sae is not None
        baseline = self._score_single_text(text, collect_all_feature_totals=True, include_logits=True)
        delta = self.sae.decoder_vector(self.layer, self.feature_id) * float(steering_strength)
        steered = self._score_single_text(
            text,
            layer_output_addition=(self.layer, delta, self._normalize_steering_positions(steering_positions)),
            collect_all_feature_totals=True,
            include_logits=True,
        )
        top_logit_deltas = self._top_logit_deltas(
            baseline.get("_last_logits"),
            steered.get("_last_logits"),
        )
        non_target_feature_deltas = self._top_feature_deltas(
            baseline.get("_all_feature_totals"),
            steered.get("_all_feature_totals"),
            exclude_feature_id=self.feature_id,
        )
        return {
            "round": round_index,
            "text": text,
            "steering_strength": float(steering_strength),
            "steering_positions": steering_positions if steering_positions is not None else "all",
            "steering_reason": steering_reason,
            "baseline": self._strip_private_metrics(baseline),
            "steered": self._strip_private_metrics(steered),
            "target_total_delta": float(
                steered["feature_total_activation"] - baseline["feature_total_activation"]
            ),
            "target_peak_delta": float(
                steered["feature_peak_activation"] - baseline["feature_peak_activation"]
            ),
            "top_logit_deltas": top_logit_deltas,
            "top_non_target_feature_deltas": non_target_feature_deltas,
            "generated_comparison": None,
        }

    def _score_single_text(
        self,
        text: str,
        layer_output_addition: tuple[int, torch.Tensor, list[int] | str | None] | None = None,
        collect_all_feature_totals: bool = False,
        include_logits: bool = False,
    ) -> dict[str, Any]:
        self._ensure_runtime()
        assert self.model is not None
        assert self.sae is not None
        token_ids = self.model.tokenize_document(text)
        windows = self.model.make_windows(
            token_ids,
            metadata_mode="heuristic",
            window_len=self.model.max_context_window_tokens(),
        )
        if not windows:
            return {
                "feature_total_activation": 0.0,
                "feature_peak_activation": 0.0,
                "active_token_fraction": 0.0,
                "top_token_hits": [],
                "window_count": 0,
            }

        total_activation = 0.0
        peak_activation = 0.0
        active_tokens = 0
        total_tokens = 0
        token_hits: list[dict[str, Any]] = []
        last_logits = None
        aggregate_feature_totals = None
        if collect_all_feature_totals:
            aggregate_feature_totals = None

        for window in windows:
            outputs, _ = self.model.forward_outputs(
                window.input_ids,
                require_grad=False,
                layer_output_addition=layer_output_addition,
            )
            residual = outputs.hidden_states[self.layer + 1]
            latents = self.sae.encode_layer(self.layer, residual).squeeze(0).detach().to("cpu")
            if self.feature_id >= latents.shape[-1]:
                token_values = torch.zeros(latents.shape[0], dtype=torch.float32)
            else:
                token_values = latents[:, self.feature_id].clamp_min(0).to(torch.float32)
            total_activation += float(token_values.sum().item())
            peak_activation = max(peak_activation, float(token_values.max().item()) if token_values.numel() else 0.0)
            active_tokens += int((token_values > 0).sum().item())
            total_tokens += int(token_values.shape[0])
            token_hits.extend(self._top_token_hits_for_window(window, token_values))
            if include_logits:
                last_logits = outputs.logits[0, -1].detach().to("cpu")
            if collect_all_feature_totals:
                all_totals = latents.clamp_min(0).sum(dim=0).to(torch.float32)
                if aggregate_feature_totals is None:
                    aggregate_feature_totals = all_totals
                else:
                    aggregate_feature_totals = aggregate_feature_totals + all_totals

        token_hits = sorted(token_hits, key=lambda row: row["activation"], reverse=True)[:5]
        result = {
            "feature_total_activation": float(total_activation),
            "feature_peak_activation": float(peak_activation),
            "active_token_fraction": float(active_tokens / total_tokens) if total_tokens else 0.0,
            "top_token_hits": token_hits,
            "window_count": len(windows),
        }
        if include_logits:
            result["_last_logits"] = last_logits
        if collect_all_feature_totals:
            result["_all_feature_totals"] = aggregate_feature_totals
        return result

    def _top_token_hits_for_window(self, window: Any, token_values: torch.Tensor) -> list[dict[str, Any]]:
        if token_values.numel() == 0:
            return []
        top_n = min(5, int(token_values.shape[0]))
        values, indices = torch.topk(token_values, k=top_n)
        hits: list[dict[str, Any]] = []
        for value, index in zip(values.tolist(), indices.tolist()):
            activation = float(value)
            if activation <= 0:
                continue
            position = int(index)
            hits.append(
                {
                    "token_position": position,
                    "token": window.tokens[position],
                    "activation": activation,
                    "snippet": self._window_snippet(window.tokens, position),
                }
            )
        return hits

    def _top_logit_deltas(self, baseline_logits: torch.Tensor | None, steered_logits: torch.Tensor | None) -> dict[str, Any]:
        if baseline_logits is None or steered_logits is None or self.model is None:
            return {"positive": [], "negative": []}
        delta = (steered_logits - baseline_logits).to(torch.float32)
        top_up_values, top_up_indices = torch.topk(delta, k=min(5, delta.shape[0]))
        top_down_values, top_down_indices = torch.topk(-delta, k=min(5, delta.shape[0]))
        return {
            "positive": [
                {
                    "token_id": int(index),
                    "token": self.model.tokenizer.convert_ids_to_tokens([int(index)])[0],
                    "delta": float(value),
                }
                for value, index in zip(top_up_values.tolist(), top_up_indices.tolist())
            ],
            "negative": [
                {
                    "token_id": int(index),
                    "token": self.model.tokenizer.convert_ids_to_tokens([int(index)])[0],
                    "delta": -float(value),
                }
                for value, index in zip(top_down_values.tolist(), top_down_indices.tolist())
            ],
        }

    @staticmethod
    def _top_feature_deltas(
        baseline_totals: torch.Tensor | None,
        steered_totals: torch.Tensor | None,
        exclude_feature_id: int,
    ) -> list[dict[str, Any]]:
        if baseline_totals is None or steered_totals is None:
            return []
        delta = (steered_totals - baseline_totals).to(torch.float32)
        if 0 <= exclude_feature_id < delta.shape[0]:
            delta[int(exclude_feature_id)] = 0
        top_values, top_indices = torch.topk(delta.abs(), k=min(5, delta.shape[0]))
        rows: list[dict[str, Any]] = []
        for magnitude, feature_idx in zip(top_values.tolist(), top_indices.tolist()):
            rows.append(
                {
                    "feature_id": int(feature_idx),
                    "delta": float(delta[int(feature_idx)].item()),
                    "abs_delta": float(magnitude),
                }
            )
        return rows

    def _ensure_runtime(self) -> None:
        if self.model is not None and self.sae is not None:
            return
        cache_key = (
            self.config.model.base_model_id,
            self.config.model.scope_release,
            self.config.model.scope_width,
            self.config.env.torch_device,
            self.config.env.torch_dtype,
        )
        cached = self._RUNTIME_CACHE.get(cache_key)
        if cached is not None:
            self.model, self.sae = cached
            return
        self.model = GemmaModelAdapter(self.config)
        self.sae = GemmaScopeAdapter(self.config, device=self.model.device, dtype=self.model.dtype)
        self._RUNTIME_CACHE[cache_key] = (self.model, self.sae)

    def _build_paths(self) -> ProbeRunPaths:
        script_segment = self._slugify(self.script_id) if self.script_id else "all_scripts"
        root = (
            Path("probe_runs")
            / self.bundle_id
            / script_segment
            / f"layer_{self.layer}"
            / f"feature_{self.feature_id}"
        )
        return ProbeRunPaths(
            root=root,
            evidence_path=root / "feature_probe_evidence.json",
            rounds_path=root / "feature_probe_rounds.jsonl",
            tests_path=root / "feature_probe_tests.jsonl",
            steering_path=root / "feature_probe_steering.jsonl",
            report_path=root / "feature_probe_report.json",
            manifest_path=root / "manifest.json",
        )

    def _stopping_no_gain(self, rounds: list[dict[str, Any]]) -> bool:
        if len(rounds) < 2:
            return False
        latest = float((rounds[-1].get("round_summary") or {}).get("support_score", 0.0))
        previous = float((rounds[-2].get("round_summary") or {}).get("support_score", 0.0))
        return abs(latest - previous) < 0.05

    @staticmethod
    def _normalize_synthetic_probes(rows: Any) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for index, row in enumerate(FeatureProbingPipeline._ensure_list(rows), start=1):
            if not isinstance(row, dict):
                continue
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            normalized.append(
                {
                    "probe_id": row.get("probe_id") or f"synthetic_{index}",
                    "text": text,
                    "expected_effect": str(row.get("expected_effect", "contrast")),
                    "probe_type": str(row.get("probe_type", "synthetic")),
                    "reason": row.get("reason"),
                }
            )
        return normalized

    @staticmethod
    def _normalize_real_edits(rows: Any) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for index, row in enumerate(FeatureProbingPipeline._ensure_list(rows), start=1):
            if not isinstance(row, dict):
                continue
            source_id = str(row.get("source_id", "")).strip()
            edited_text = str(row.get("edited_text", "")).strip()
            if not source_id or not edited_text:
                continue
            normalized.append(
                {
                    "edit_id": row.get("edit_id") or f"real_edit_{index}",
                    "source_id": source_id,
                    "edited_text": edited_text,
                    "expected_effect": str(row.get("expected_effect", "contrast")),
                    "edit_type": str(row.get("edit_type", "rewrite")),
                    "reason": row.get("reason"),
                }
            )
        return normalized

    @staticmethod
    def _normalize_steering_positions(value: Any) -> list[int] | str | None:
        if value in {None, "all", "last"}:
            return value
        if isinstance(value, list):
            return [int(item) for item in value]
        return "all"

    @staticmethod
    def _compact_steering_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "round": row.get("round"),
                "steering_strength": row.get("steering_strength"),
                "target_total_delta": row.get("target_total_delta"),
                "target_peak_delta": row.get("target_peak_delta"),
                "top_logit_deltas": row.get("top_logit_deltas"),
            }
            for row in rows[:6]
        ]

    @staticmethod
    def _top_probe_rows(rows: list[dict[str, Any]], expected: str) -> list[dict[str, Any]]:
        filtered = [
            row
            for row in rows
            if row.get("test_kind") == "synthetic_probe" and row.get("expected_effect") == expected
        ]
        ordered = sorted(filtered, key=lambda row: float(row.get("feature_total_activation", 0.0)), reverse=True)
        if expected == "negative":
            ordered = sorted(filtered, key=lambda row: float(row.get("feature_total_activation", 0.0)))
        return [
            {
                "probe_id": row.get("probe_id"),
                "text": row.get("text"),
                "feature_total_activation": row.get("feature_total_activation"),
                "feature_peak_activation": row.get("feature_peak_activation"),
            }
            for row in ordered[:5]
        ]

    @staticmethod
    def _top_edit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        edits = [row for row in rows if row.get("test_kind") == "real_edit"]
        ordered = sorted(edits, key=lambda row: abs(float(row.get("comparison_delta", 0.0))), reverse=True)
        return [
            {
                "probe_id": row.get("probe_id"),
                "source_id": row.get("source_id"),
                "edited_text": row.get("edited_text"),
                "comparison_delta": row.get("comparison_delta"),
                "expected_effect": row.get("expected_effect"),
            }
            for row in ordered[:5]
        ]

    @staticmethod
    def _bag_of_words(text: str) -> dict[str, float]:
        counts: dict[str, float] = {}
        for token in re.findall(r"[A-Za-z0-9_']+", str(text).lower()):
            counts[token] = counts.get(token, 0.0) + 1.0
        return counts

    @staticmethod
    def _cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
        if not left or not right:
            return 0.0
        dot = sum(value * right.get(token, 0.0) for token, value in left.items())
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        if left_norm <= 0 or right_norm <= 0:
            return 0.0
        return float(dot / (left_norm * right_norm))

    @staticmethod
    def _strip_private_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in metrics.items() if not key.startswith("_")}

    @staticmethod
    def _window_snippet(tokens: list[str], position: int, radius: int = 3) -> str:
        start = max(0, position - radius)
        end = min(len(tokens), position + radius + 1)
        return " ".join(tokens[start:end])

    @staticmethod
    def _normalize_source_text(text: Any) -> str:
        return str(text or "").replace("\n", " ").strip()

    @staticmethod
    def _ensure_list(value: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        if value is None or value == "":
            return []
        return [value]

    @staticmethod
    def _slugify(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "default"

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    rows.append(json.loads(stripped))
        return rows

    @staticmethod
    def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _dolma_context_text(context: dict[str, Any]) -> str:
        sentence_snippet = ((context.get("top_sentence_snippets") or [None])[0] or {}).get("text")
        if sentence_snippet:
            return str(sentence_snippet).strip()
        span_tokens = (((context.get("top_span_snippets") or [None])[0] or {}).get("tokens") or [])
        if span_tokens:
            return " ".join(str(token) for token in span_tokens).strip()
        token_snippet = (((context.get("top_token_snippets") or [None])[0] or {}).get("snippet_tokens") or [])
        if token_snippet:
            return " ".join(str(token) for token in token_snippet).strip()
        return str(context.get("window_text") or "").strip()
