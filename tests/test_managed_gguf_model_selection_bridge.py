"""CODE DEFECT regression: managed llama_cpp runs backed by a resolved local
GGUF artifact (Stage 3B.3E ``gguf_artifacts``/``gguf_root``) must be able to
select the model they just materialised.

``_resolve_managed_llama_artifact`` accepts a requested ``--models`` ref
(often an HF/Ollama-shaped string like ``hf.co/org/repo:Q8_0``) that resolves
to exactly one local GGUF file. The managed llama-server then serves that
file under its own reported name/id via ``/v1/models`` -- never the original
ref. ``_resolve_model_selection`` -> ``resolve_exact_models`` exact/fuzzy
matches the requested ref against ``client.tags()`` and unconditionally
rejects it as "not an installed model", even though materialisation already
proved the ref resolves to exactly the model now being served.

The fix (``cli._apply_managed_owned_model_selection``) sets
``args._selected_models`` -- the existing pre-resolution seam read at the top
of ``cmd_run`` -- to the single served name, but only for the exact shape the
boundaries require: an owned managed llama_cpp spawn, a single requested
ref, and that ref matching the artifact resolution's own ``model_ref``. It
must never fire for Ollama, external llama-server reuse, multi-model
selection, ``--select``, or an unresolved/unknown ref.
"""
from __future__ import annotations

import json

import pytest

from llm_modelbench import cli
from llm_modelbench.config import Config
from llm_modelbench.selection import resolve_exact_models as _real_resolve_exact_models

from test_stage3b3d_cmd_run_wiring import (
    _owned_outcome,
    _reuse_outcome,
    _run_args,
    _wire_full_run,
)


def _owned_outcome_with_artifact(model_ref="hf.co/org/repo:Q8_0", **kw):
    outcome = _owned_outcome(**kw)
    from dataclasses import replace

    return replace(
        outcome,
        artifact_resolution={
            "status": "resolved",
            "model_ref": model_ref,
            "owned_placement": True,
            "resolved_path": "/models/repo-Q8_0.gguf",
            "verified_sha256": "e" * 64,
        },
    )


def _use_real_selection(monkeypatch):
    """_wire_full_run stubs _resolve_model_selection to a fixed success value
    (it is not the seam these tests exercise); restore the real
    exact-match/fuzzy-match implementation so a rejected ref actually
    raises."""
    from llm_modelbench.selection import parse_models_spec, select_models

    def _real(args, client):
        installed = [row.get("name") for row in client.tags() if row.get("name")]
        requested = parse_models_spec(getattr(args, "models", None))
        if requested is not None:
            return _real_resolve_exact_models(requested, installed)
        if getattr(args, "select", False):
            return select_models(installed)
        return None

    monkeypatch.setattr(cli, "_resolve_model_selection", _real)


class _ServedOneModelClient:
    endpoint = "http://127.0.0.1:9099"

    def __init__(self, served_name="repo-Q8_0"):
        self._served_name = served_name

    def tags(self):
        return [{"name": self._served_name}]


def test_managed_owned_ref_is_accepted_before_and_maps_to_served_name(monkeypatch, tmp_path):
    """The single requested ref that resolved to the artifact must select the
    run instead of raising "not an installed model", and the model actually
    passed to runner.run must be the server-reported name, not the ref
    (identity_run wiring, tag_rows, and client._model() all key off the
    served name -- see cli.py:1083 and llama_cpp.py:_model())."""
    ref = "hf.co/org/repo:Q8_0"
    outcome = _owned_outcome_with_artifact(model_ref=ref)
    seen = {}

    def _run(client, cfg, **kw):
        seen["selected_models"] = kw["selected_models"]
        kw["out_dir"].mkdir(parents=True, exist_ok=True)
        (kw["out_dir"] / "raw_results.jsonl").write_text("", encoding="utf-8")
        return kw["out_dir"]

    _wire_full_run(monkeypatch, outcome, run_impl=_run)
    monkeypatch.setattr(cli, "_resolve_model_selection",
                        lambda args, client: (_ for _ in ()).throw(
                            AssertionError("must not fall through to exact-match selection")))
    monkeypatch.setattr(cli, "_client_for_materialised_endpoint",
                        lambda endpoint, cfg, *, backend: _ServedOneModelClient(), raising=False)

    cli.cmd_run(_run_args(tmp_path, "--models", ref), Config())
    assert seen["selected_models"] == ["repo-Q8_0"]


def test_unknown_ref_still_fails(monkeypatch, tmp_path):
    """A ref that does not match the artifact resolution's own model_ref
    (materialisation never resolved it, or a different ref was configured)
    must still hit the ordinary exact-selection failure, never be waved
    through."""
    outcome = _owned_outcome_with_artifact(model_ref="hf.co/org/repo:Q8_0")

    def _run(client, cfg, **kw):  # pragma: no cover - must not be reached
        raise AssertionError("must not run")

    _wire_full_run(monkeypatch, outcome, run_impl=_run)
    _use_real_selection(monkeypatch)
    monkeypatch.setattr(cli, "_client_for_materialised_endpoint",
                        lambda endpoint, cfg, *, backend: _ServedOneModelClient(), raising=False)

    with pytest.raises(ValueError, match="not an installed model"):
        cli.cmd_run(_run_args(tmp_path, "--models", "hf.co/other/unrelated:Q4_0"), Config())


def test_ambiguous_gguf_root_still_fails(monkeypatch, tmp_path):
    """An artifact resolution that did not resolve (e.g. ambiguous gguf_root,
    status != 'resolved') must never trigger the bridge -- selection falls
    through to the ordinary exact-match path and fails closed."""
    outcome = _owned_outcome(**{})
    from dataclasses import replace

    outcome = replace(outcome, artifact_resolution={
        "status": "ambiguous", "model_ref": "shared-name", "owned_placement": False,
    })

    def _run(client, cfg, **kw):  # pragma: no cover - must not be reached
        raise AssertionError("must not run")

    _wire_full_run(monkeypatch, outcome, run_impl=_run)
    _use_real_selection(monkeypatch)
    monkeypatch.setattr(cli, "_client_for_materialised_endpoint",
                        lambda endpoint, cfg, *, backend: _ServedOneModelClient(), raising=False)

    with pytest.raises(ValueError, match="not an installed model"):
        cli.cmd_run(_run_args(tmp_path, "--models", "shared-name"), Config())


def test_ollama_exact_selection_behaviour_unchanged(monkeypatch, tmp_path):
    """The bridge must never fire on a reuse-only outcome (Ollama, or
    llama_cpp with no resolved artifact) -- normal exact-match selection
    against client.tags() proceeds untouched."""
    outcome = _reuse_outcome()
    seen = {}

    def _run(client, cfg, **kw):
        seen["selected_models"] = kw["selected_models"]
        kw["out_dir"].mkdir(parents=True, exist_ok=True)
        (kw["out_dir"] / "raw_results.jsonl").write_text("", encoding="utf-8")
        return kw["out_dir"]

    _wire_full_run(monkeypatch, outcome, run_impl=_run)
    monkeypatch.setattr(cli, "_resolve_model_selection", lambda args, client: ["m:latest"])

    cli.cmd_run(_run_args(tmp_path, "--models", "m:latest"), Config())
    assert seen["selected_models"] == ["m:latest"]


def test_multi_model_selection_is_not_bridged(monkeypatch, tmp_path):
    """A multi-model --models request must never be redirected through the
    single-served-model bridge, even on an owned managed run (the artifact
    resolver itself only ever resolves a single explicit ref -- Stage 3B.3E
    owner decision 3B.3E-OD1 -- so a multi-model request cannot have produced
    an owned_placement=True/resolved snapshot in practice, but the bridge
    must not assume that and must check request shape defensively)."""
    outcome = _owned_outcome_with_artifact(model_ref="a")

    def _run(client, cfg, **kw):  # pragma: no cover
        raise AssertionError("must not run")

    _wire_full_run(monkeypatch, outcome, run_impl=_run)
    _use_real_selection(monkeypatch)
    monkeypatch.setattr(cli, "_client_for_materialised_endpoint",
                        lambda endpoint, cfg, *, backend: _ServedOneModelClient(), raising=False)

    with pytest.raises(ValueError, match="not an installed model"):
        cli.cmd_run(_run_args(tmp_path, "--models", "a;b"), Config())


def test_cleanup_still_occurs_when_selection_raises_after_managed_spawn(monkeypatch, tmp_path):
    """Regression for the real managed-smoke failure this patch fixes: before
    the bridge existed, ValueError from _resolve_model_selection after a
    successful managed spawn still had to clean up the owned runtime. Confirm
    that remains true for the unresolved-ref path (bridge does not fire,
    ordinary exact-match raises) -- the owned llama-server must still be torn
    down exactly once, with structured evidence persisted."""
    outcome = _owned_outcome_with_artifact(model_ref="hf.co/org/repo:Q8_0")

    def _run(client, cfg, **kw):  # pragma: no cover
        raise AssertionError("must not run")

    _wire_full_run(monkeypatch, outcome, run_impl=_run)
    _use_real_selection(monkeypatch)
    monkeypatch.setattr(cli, "_client_for_materialised_endpoint",
                        lambda endpoint, cfg, *, backend: _ServedOneModelClient(), raising=False)

    with pytest.raises(ValueError, match="not an installed model"):
        cli.cmd_run(_run_args(tmp_path, "--models", "hf.co/other/unrelated:Q4_0"), Config())
    assert outcome.controller.cleanup_calls == 1
    ev = json.loads((tmp_path / "r" / "materialisation_evidence.json").read_text())
    assert ev["cleanup"]["observed"] is True
    assert ev["benchmark_completed"] is False
