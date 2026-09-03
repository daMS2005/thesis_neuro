"""Standard-library HTTP server behind the transcript and feature audit dashboard."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from thesis_neuro.audit_data import (
    AuditBundle,
    build_feature_lookup,
    build_focus_features,
    build_script_audit_view,
    build_token_tooltip,
    find_script_default,
    load_audit_bundle,
    resolve_audit_paths,
)
from thesis_neuro.paths import default_config_path, output_root, repository_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="thesis-neuro-audit")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--analysis-dir", default="outputs/default-run")
    parser.add_argument("--transcript-dir", default="outputs/default-run")
    parser.add_argument("--dolma-dir", default="outputs/default-run")
    parser.add_argument(
        "--bundle",
        action="append",
        default=[],
        help="Repeated bundle spec: bundle_id|label|analysis_dir|transcript_dir|dolma_dir",
    )
    parser.add_argument("--default-script", default="shapesphysical")
    parser.add_argument("--default-layer", type=int, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    return parser.parse_args()


@dataclass(slots=True)
class AuditBundleState:
    bundle_id: str
    label: str
    bundle: AuditBundle
    default_script_id: str
    default_layer: int


@dataclass(slots=True)
class AuditServerState:
    repo_root: Path
    bundles: dict[str, AuditBundleState]
    default_bundle_id: str
    script_view_cache: dict[tuple[str, str, int], dict[str, Any]]
    probe_jobs: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get_bundle_state(self, bundle_id: str | None = None) -> AuditBundleState:
        resolved_bundle_id = bundle_id or self.default_bundle_id
        bundle_state = self.bundles.get(resolved_bundle_id)
        if bundle_state is None:
            raise ValueError(f"Unknown bundle_id: {resolved_bundle_id}")
        return bundle_state

    def get_script_view(self, bundle_id: str, script_id: str, layer: int) -> dict[str, Any]:
        key = (bundle_id, script_id, int(layer))
        if key not in self.script_view_cache:
            bundle_state = self.get_bundle_state(bundle_id)
            self.script_view_cache[key] = build_script_audit_view(
                bundle_state.bundle,
                script_id=script_id,
                layer=int(layer),
            )
        return self.script_view_cache[key]


class LazyAuditServerState:
    def __init__(self, args: argparse.Namespace):
        self._args = args
        self._lock = threading.Lock()
        self._state: AuditServerState | None = None
        self._error: str | None = None
        self._loading = False

    def start_loading(self) -> None:
        with self._lock:
            if self._state is not None or self._loading:
                return
            self._loading = True
        thread = threading.Thread(target=self._load, daemon=True)
        thread.start()

    def _load(self) -> None:
        try:
            state = build_server_state(self._args)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
                self._loading = False
            return
        with self._lock:
            self._state = state
            self._loading = False

    def status_payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "ready": self._state is not None,
                "loading": self._loading,
                "error": self._error,
            }

    def require_state(self) -> AuditServerState:
        self.start_loading()
        with self._lock:
            if self._state is not None:
                return self._state
            if self._error is not None:
                raise RuntimeError(self._error)
        raise RuntimeError("Audit bundles are still loading. Try again in a few seconds.")


def build_server_state(args: argparse.Namespace) -> AuditServerState:
    repo_root = Path(args.repo_root).expanduser().resolve() if args.repo_root else repository_root()
    bundle_specs = _bundle_specs_from_args(args)
    bundles: dict[str, AuditBundleState] = {}
    for spec in bundle_specs:
        paths = resolve_audit_paths(
            repo_root=args.repo_root,
            analysis_dir=spec["analysis_dir"],
            transcript_dir=spec["transcript_dir"],
            dolma_dir=spec["dolma_dir"],
        )
        bundle = load_audit_bundle(paths)
        if not bundle.scripts:
            raise ValueError(f"No transcript scripts found for bundle {spec['bundle_id']}.")
        default_script = find_script_default(bundle, preferred_script=args.default_script) or bundle.scripts[0]
        default_script_id = str(default_script["script_id"])
        layers = list(default_script.get("layers") or [])
        if not layers:
            raise ValueError(f"No layers available for script {default_script_id} in bundle {spec['bundle_id']}")
        default_layer = int(args.default_layer) if args.default_layer in layers else int(layers[-1])
        bundles[spec["bundle_id"]] = AuditBundleState(
            bundle_id=spec["bundle_id"],
            label=spec["label"],
            bundle=bundle,
            default_script_id=default_script_id,
            default_layer=default_layer,
        )
    if not bundles:
        raise ValueError("No audit bundles configured.")
    return AuditServerState(
        repo_root=repo_root,
        bundles=bundles,
        default_bundle_id=next(iter(bundles)),
        script_view_cache={},
    )


def build_handler(lazy_state: LazyAuditServerState):
    class AuditRequestHandler(BaseHTTPRequestHandler):
        server_version = "ThesisNeuroAudit/0.1"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(INDEX_HTML)
                return
            if parsed.path == "/static/audit.css":
                self._send_asset(AUDIT_CSS, "text/css; charset=utf-8")
                return
            if parsed.path == "/static/audit.js":
                self._send_asset(AUDIT_JS, "text/javascript; charset=utf-8")
                return
            if parsed.path == "/api/health":
                self._send_json({"ok": True, **lazy_state.status_payload()})
                return
            if parsed.path == "/api/meta":
                self._send_json(_meta_payload(lazy_state.require_state()))
                return
            if parsed.path == "/api/script-view":
                payload = _script_view_payload(lazy_state.require_state(), parse_qs(parsed.query))
                self._send_json(payload)
                return
            if parsed.path == "/api/window":
                payload = _window_payload(lazy_state.require_state(), parse_qs(parsed.query))
                self._send_json(payload)
                return
            if parsed.path == "/api/focus":
                payload = _focus_payload(lazy_state.require_state(), parse_qs(parsed.query))
                self._send_json(payload)
                return
            if parsed.path == "/api/feature":
                payload = _feature_payload(lazy_state.require_state(), parse_qs(parsed.query))
                self._send_json(payload)
                return
            if parsed.path == "/api/probe-status":
                payload = _probe_status_payload(lazy_state.require_state(), parse_qs(parsed.query))
                self._send_json(payload)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/api/probe-start":
                payload = self._read_json_body()
                response = _probe_start_payload(lazy_state.require_state(), payload)
                self._send_json(response)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_html(self, html_text: str) -> None:
            self._send_asset(html_text, "text/html; charset=utf-8")

        def _send_asset(self, text: str, content_type: str) -> None:
            body = text.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: Any) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length) if length > 0 else b"{}"
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))

    return AuditRequestHandler


def _bundle_specs_from_args(args: argparse.Namespace) -> list[dict[str, str | None]]:
    if not args.bundle:
        return [
            {
                "bundle_id": "default",
                "label": "Default",
                "analysis_dir": args.analysis_dir,
                "transcript_dir": args.transcript_dir,
                "dolma_dir": args.dolma_dir,
            }
        ]
    specs: list[dict[str, str | None]] = []
    for raw_value in args.bundle:
        parts = [part.strip() for part in str(raw_value).split("|")]
        if len(parts) != 5:
            raise ValueError(
                "Invalid --bundle spec. Expected bundle_id|label|analysis_dir|transcript_dir|dolma_dir"
            )
        bundle_id, label, analysis_dir, transcript_dir, dolma_dir = parts
        specs.append(
            {
                "bundle_id": bundle_id,
                "label": label or bundle_id,
                "analysis_dir": analysis_dir,
                "transcript_dir": transcript_dir or analysis_dir,
                "dolma_dir": dolma_dir or analysis_dir,
            }
        )
    return specs


def _meta_payload(state: AuditServerState) -> dict[str, Any]:
    default_bundle = state.get_bundle_state()
    return {
        "default_bundle_id": state.default_bundle_id,
        "default_script_id": default_bundle.default_script_id,
        "default_layer": default_bundle.default_layer,
        "bundles": [
            {
                "bundle_id": bundle_state.bundle_id,
                "label": bundle_state.label,
                "default_script_id": bundle_state.default_script_id,
                "default_layer": bundle_state.default_layer,
                "scripts": bundle_state.bundle.scripts,
                "paths": {
                    "analysis_dir": str(bundle_state.bundle.paths.analysis_dir),
                    "transcript_dir": str(bundle_state.bundle.paths.transcript_dir)
                    if bundle_state.bundle.paths.transcript_dir
                    else None,
                    "dolma_dir": str(bundle_state.bundle.paths.dolma_dir)
                    if bundle_state.bundle.paths.dolma_dir
                    else None,
                },
            }
            for bundle_state in state.bundles.values()
        ],
    }


def _script_view_payload(state: AuditServerState, query: dict[str, list[str]]) -> dict[str, Any]:
    bundle_state = state.get_bundle_state(_required_str(query, "bundle_id", default=state.default_bundle_id))
    script_id = _required_str(query, "script_id", default=bundle_state.default_script_id)
    script_meta = next((script for script in bundle_state.bundle.scripts if script["script_id"] == script_id), None)
    if script_meta is None:
        raise ValueError(f"Unknown script_id: {script_id}")
    layer = _required_int(query, "layer", default=bundle_state.default_layer)
    raw_view = state.get_script_view(bundle_state.bundle_id, script_id, layer)
    windows: list[dict[str, Any]] = []
    for window in raw_view.get("windows", []):
        windows.append(
            {
                "sample_id": window.get("sample_id"),
                "script_id": window.get("script_id"),
                "text": window.get("text", ""),
                "window_start": window.get("window_start"),
                "window_end": window.get("window_end"),
                "provenance": window.get("provenance", {}),
                "sentence_count": len(window.get("sentences", [])),
                "token_count": len(window.get("token_details", [])),
                "window_feature_count": len(window.get("window_features", [])),
            }
        )
    return {
        "bundle_id": bundle_state.bundle_id,
        "script": raw_view.get("script"),
        "top_script_features": raw_view.get("top_script_features", []),
        "windows": windows,
    }


def _window_payload(state: AuditServerState, query: dict[str, list[str]]) -> dict[str, Any]:
    bundle_state = state.get_bundle_state(_required_str(query, "bundle_id", default=state.default_bundle_id))
    script_id = _required_str(query, "script_id", default=bundle_state.default_script_id)
    layer = _required_int(query, "layer", default=bundle_state.default_layer)
    sample_id = _required_str(query, "sample_id")
    raw_view = state.get_script_view(bundle_state.bundle_id, script_id, layer)
    window = next((item for item in raw_view.get("windows", []) if str(item.get("sample_id")) == sample_id), None)
    if window is None:
        raise ValueError(f"Unknown sample_id: {sample_id}")
    token_details = []
    for token in window.get("token_details", []):
        token_details.append(
            {
                **token,
                "tooltip": build_token_tooltip(bundle_state.bundle, layer, token),
                "signal_total": float(
                    sum(float(item.get("activation", 0.0)) for item in token.get("latent_activations", []))
                ),
            }
        )
    return {
        "window": {
            **window,
            "token_details": token_details,
        }
    }


def _focus_payload(state: AuditServerState, query: dict[str, list[str]]) -> dict[str, Any]:
    bundle_state = state.get_bundle_state(_required_str(query, "bundle_id", default=state.default_bundle_id))
    script_id = _required_str(query, "script_id", default=bundle_state.default_script_id)
    layer = _required_int(query, "layer", default=bundle_state.default_layer)
    sample_id = _required_str(query, "sample_id")
    mode = _required_str(query, "mode", default="token")
    lens = _required_str(query, "lens", default="strongest")
    token_position = _optional_int(query, "token_position")
    sentence_id = _optional_int(query, "sentence_id")
    span_start = _optional_int(query, "span_start")
    span_end = _optional_int(query, "span_end")
    feature_filter = _optional_int(query, "feature_filter")

    script_view = state.get_script_view(bundle_state.bundle_id, script_id, layer)
    window = next((item for item in script_view.get("windows", []) if str(item["sample_id"]) == sample_id), None)
    if window is None:
        raise ValueError(f"Unknown sample_id: {sample_id}")

    focus_rows = build_focus_features(
        bundle_state.bundle,
        layer=layer,
        script_view=script_view,
        window=window,
        mode=mode,
        lens=lens,
        token_position=token_position,
        sentence_id=sentence_id,
        span_start=span_start,
        span_end=span_end,
        feature_filter=feature_filter,
    )
    return {
        "rows": [_compact_focus_row(row) for row in focus_rows],
    }


def _feature_payload(state: AuditServerState, query: dict[str, list[str]]) -> dict[str, Any]:
    bundle_state = state.get_bundle_state(_required_str(query, "bundle_id", default=state.default_bundle_id))
    script_id = _required_str(query, "script_id", default=bundle_state.default_script_id)
    layer = _required_int(query, "layer", default=bundle_state.default_layer)
    feature_id = _required_int(query, "feature_id")
    script_view = state.get_script_view(bundle_state.bundle_id, script_id, layer)
    row = build_feature_lookup(bundle_state.bundle, int(layer), script_view, int(feature_id))
    return {"row": _compact_focus_row(row)}


def _probe_status_payload(state: AuditServerState, query: dict[str, list[str]]) -> dict[str, Any]:
    bundle_state = state.get_bundle_state(_required_str(query, "bundle_id", default=state.default_bundle_id))
    script_id = _required_str(query, "script_id", default=bundle_state.default_script_id)
    layer = _required_int(query, "layer", default=bundle_state.default_layer)
    feature_id = _required_int(query, "feature_id")
    return _probe_status_for_feature(
        state=state,
        bundle_state=bundle_state,
        script_id=script_id,
        layer=layer,
        feature_id=feature_id,
    )


def _probe_start_payload(state: AuditServerState, payload: dict[str, Any]) -> dict[str, Any]:
    bundle_state = state.get_bundle_state(str(payload.get("bundle_id") or state.default_bundle_id))
    script_id = str(payload.get("script_id") or bundle_state.default_script_id)
    layer = int(payload.get("layer") or bundle_state.default_layer)
    feature_id = int(payload.get("feature_id"))
    status = _probe_status_for_feature(
        state=state,
        bundle_state=bundle_state,
        script_id=script_id,
        layer=layer,
        feature_id=feature_id,
    )
    if status["running"] or status["has_report"]:
        return status

    probe_paths = _probe_paths(bundle_state, script_id=script_id, layer=layer, feature_id=feature_id)
    config_path = default_config_path()
    python_bin = Path(sys.executable)
    probe_paths["root"].mkdir(parents=True, exist_ok=True)
    command = [
        str(python_bin),
        "-m",
        "thesis_neuro",
        "--config",
        str(config_path),
        "--env-file",
        str(state.repo_root / ".env"),
        "probe-feature",
        "--analysis-dir",
        str(bundle_state.bundle.paths.analysis_dir),
        "--transcript-dir",
        str(bundle_state.bundle.paths.transcript_dir or bundle_state.bundle.paths.analysis_dir),
        "--dolma-dir",
        str(bundle_state.bundle.paths.dolma_dir or bundle_state.bundle.paths.analysis_dir),
        "--alignment-path",
        str(bundle_state.bundle.paths.analysis_dir / "feature_alignment.jsonl"),
        "--layer",
        str(layer),
        "--feature-id",
        str(feature_id),
        "--script-id",
        script_id,
    ]
    env = os.environ.copy()
    with probe_paths["log_path"].open("ab") as handle:
        process = subprocess.Popen(  # noqa: S603
            command,
            cwd=state.repo_root,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    job_key = _probe_job_key(bundle_state.bundle_id, script_id, layer, feature_id)
    state.probe_jobs[job_key] = {
        "pid": process.pid,
        "started_at": time.time(),
        "log_path": str(probe_paths["log_path"]),
        "command": command,
    }
    return _probe_status_for_feature(
        state=state,
        bundle_state=bundle_state,
        script_id=script_id,
        layer=layer,
        feature_id=feature_id,
    )


def _probe_status_for_feature(
    state: AuditServerState,
    bundle_state: AuditBundleState,
    script_id: str,
    layer: int,
    feature_id: int,
) -> dict[str, Any]:
    probe_paths = _probe_paths(bundle_state, script_id=script_id, layer=layer, feature_id=feature_id)
    job_key = _probe_job_key(bundle_state.bundle_id, script_id, layer, feature_id)
    job = state.probe_jobs.get(job_key) or {}
    pid = int(job["pid"]) if job.get("pid") else None
    running = bool(pid and _pid_is_running(pid))
    if pid and not running:
        state.probe_jobs.pop(job_key, None)
    report_data = _read_json_if_exists(probe_paths["report_path"])
    manifest_data = _read_json_if_exists(probe_paths["manifest_path"])
    evidence_data = _read_json_if_exists(probe_paths["evidence_path"])
    return {
        "bundle_id": bundle_state.bundle_id,
        "script_id": script_id,
        "layer": layer,
        "feature_id": feature_id,
        "running": running,
        "pid": pid if running else None,
        "started_at": job.get("started_at"),
        "has_report": probe_paths["report_path"].exists(),
        "has_manifest": probe_paths["manifest_path"].exists(),
        "has_evidence": probe_paths["evidence_path"].exists(),
        "paths": {
            "root": str(probe_paths["root"]),
            "log": str(probe_paths["log_path"]),
            "report": str(probe_paths["report_path"]),
            "manifest": str(probe_paths["manifest_path"]),
            "evidence": str(probe_paths["evidence_path"]),
            "tests": str(probe_paths["tests_path"]),
            "rounds": str(probe_paths["rounds_path"]),
            "steering": str(probe_paths["steering_path"]),
        },
        "report_summary": _probe_report_summary(report_data),
        "manifest_summary": _probe_manifest_summary(manifest_data),
        "evidence_summary": _probe_evidence_summary(evidence_data),
        "log_tail": _tail_text(probe_paths["log_path"], lines=20),
    }


def _probe_paths(
    bundle_state: AuditBundleState,
    script_id: str | None,
    layer: int,
    feature_id: int,
) -> dict[str, Path]:
    script_segment = _slugify(script_id) if script_id else "all_scripts"
    root = (
        output_root()
        / "probe_runs"
        / bundle_state.bundle.paths.analysis_dir.name
        / script_segment
        / f"layer_{int(layer)}"
        / f"feature_{int(feature_id)}"
    )
    return {
        "root": root,
        "log_path": root / "dashboard_probe.log",
        "report_path": root / "feature_probe_report.json",
        "manifest_path": root / "manifest.json",
        "evidence_path": root / "feature_probe_evidence.json",
        "tests_path": root / "feature_probe_tests.jsonl",
        "rounds_path": root / "feature_probe_rounds.jsonl",
        "steering_path": root / "feature_probe_steering.jsonl",
    }


def _probe_job_key(bundle_id: str, script_id: str, layer: int, feature_id: int) -> str:
    return f"{bundle_id}|{_slugify(script_id)}|{int(layer)}|{int(feature_id)}"


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_") or "default"


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _probe_report_summary(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    return {
        "feature_label": payload.get("feature_label"),
        "one_sentence_summary": payload.get("one_sentence_summary"),
        "confidence_label": payload.get("confidence_label"),
        "support_score": payload.get("support_score"),
    }


def _probe_manifest_summary(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    return {
        "rounds_written": payload.get("rounds_written"),
        "tests_written": payload.get("tests_written"),
        "steering_rows_written": payload.get("steering_rows_written"),
        "report_path": payload.get("report_path"),
    }


def _probe_evidence_summary(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    return {
        "transcript_examples": len(payload.get("top_transcript_examples") or []),
        "dolma_contexts": len(payload.get("top_dolma_contexts") or []),
        "alignment_rows": len(payload.get("alignment_summary") or []),
        "judge_available": bool(payload.get("judge_summary") or payload.get("judge_label")),
    }


def _tail_text(path: Path, lines: int = 20) -> list[str]:
    if not path.exists():
        return []
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:  # noqa: BLE001
        return []
    return content[-lines:]


def _compact_focus_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer": row.get("layer"),
        "feature_id": row.get("feature_id"),
        "label": row.get("label"),
        "feature_type": row.get("feature_type"),
        "confidence": row.get("confidence"),
        "summary": row.get("summary"),
        "transcript_rationale": row.get("transcript_rationale"),
        "judge_label": row.get("judge_label"),
        "judge_summary": row.get("judge_summary"),
        "judge_evidence_for": row.get("judge_evidence_for", []),
        "judge_evidence_against": row.get("judge_evidence_against", []),
        "judge_uncertainty": row.get("judge_uncertainty"),
        "judge_follow_up": row.get("judge_follow_up", []),
        "transcript_relevance_rank": row.get("transcript_relevance_rank"),
        "transcript_relevance_score": row.get("transcript_relevance_score"),
        "coverage": row.get("coverage", []),
        "has_judge": row.get("has_judge"),
        "has_alignment": row.get("has_alignment"),
        "has_dolma": row.get("has_dolma"),
        "judge_coverage_status": row.get("judge_coverage_status"),
        "distinctiveness": row.get("distinctiveness"),
        "source_note": row.get("source_note"),
        "local_metrics": row.get("local_metrics", {}),
        "top_transcript_examples": row.get("top_transcript_examples", []),
        "top_dolma_contexts": row.get("top_dolma_contexts", []),
        "top_correlated_features": row.get("top_correlated_features", []),
        "alignment_summary": row.get("alignment_summary", []),
        "transcript_support": row.get("transcript_support", {}),
        "transcript_metrics": row.get("transcript_metrics", {}),
        "script_token_hits": row.get("script_token_hits", []),
        "script_sentence_hits": row.get("script_sentence_hits", []),
        "script_window_hits": row.get("script_window_hits", []),
    }


def _required_str(query: dict[str, list[str]], name: str, default: str | None = None) -> str:
    values = query.get(name)
    if values and values[0]:
        return str(values[0])
    if default is not None:
        return default
    raise ValueError(f"Missing query parameter: {name}")


def _required_int(query: dict[str, list[str]], name: str, default: int | None = None) -> int:
    values = query.get(name)
    if values and values[0]:
        return int(values[0])
    if default is not None:
        return int(default)
    raise ValueError(f"Missing query parameter: {name}")


def _optional_int(query: dict[str, list[str]], name: str) -> int | None:
    values = query.get(name)
    if not values or values[0] == "":
        return None
    return int(values[0])


def main() -> None:
    args = parse_args()
    lazy_state = LazyAuditServerState(args)
    lazy_state.start_loading()
    handler = build_handler(lazy_state)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    host = args.host
    if host == "0.0.0.0":
        host = "127.0.0.1"
    print(f"Transcript audit web app serving on http://{host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


INDEX_HTML = resources.files("thesis_neuro.static").joinpath("audit.html").read_text(encoding="utf-8")
AUDIT_CSS = resources.files("thesis_neuro.static").joinpath("audit.css").read_text(encoding="utf-8")
AUDIT_JS = resources.files("thesis_neuro.static").joinpath("audit.js").read_text(encoding="utf-8")


if __name__ == "__main__":
    main()
