"""Anvil Stage 3.1: ``ModelCard`` golden-file and evidence-trust tests.

Per the architecture proposal's test strategy (§8, "3.1"): golden-file
comparison of pre/post ``ModelCard`` output for a fixed, corrected
``master_summary.json`` fixture, plus an explicit regression that a row with
declared ``embedding`` but measured ``text``/``tools`` produces the measured
families -- the same answer in ``ModelCard`` as in ``master_summary.json``
itself (Stage 3.0A already made ``master_summary.json``'s ``families``
measured-only; this proves ``ModelCard`` did not introduce a second,
independent derivation that could disagree with it).
"""
import json

import pytest

from llm_modelbench import rankings
from llm_modelbench.capabilities import CAPABILITY_SCHEMA_VERSION
from llm_modelbench.evidence import EvidenceTrustClass
from llm_modelbench.human_validation import HumanCorrelation, HumanValidationStatus
from llm_modelbench.model_card import ModelCard, ModelCardError
from llm_modelbench.model_cards import build_card, generate_model_cards


def _minimal_card(**overrides):
    kwargs = dict(
        model="m", digest="d1", artifact=None, identity={}, quality={}, limits={},
        long_context={}, kv_compatibility={}, evidence_warnings=[],
        evidence_trust_class=EvidenceTrustClass.CANONICAL_COMPATIBLE,
    )
    kwargs.update(overrides)
    return ModelCard(**kwargs)


def test_model_card_rejects_non_enum_trust_class():
    with pytest.raises(ModelCardError):
        _minimal_card(evidence_trust_class="canonical_compatible")  # type: ignore[arg-type]


def test_model_card_rejects_non_enum_human_validation_status():
    with pytest.raises(ModelCardError):
        _minimal_card(human_validation_status="validated")  # type: ignore[arg-type]


def test_model_card_rejects_correlation_without_a_validation_status():
    correlation = HumanCorrelation(
        metric="pearson_r", value=0.8, n=20, rubric_version="v1", evidence_ref="ref-1",
    )
    with pytest.raises(ModelCardError):
        _minimal_card(human_correlation=correlation)


def test_model_card_accepts_correlation_alongside_provisional_status():
    correlation = HumanCorrelation(
        metric="pearson_r", value=0.8, n=20, rubric_version="v1", evidence_ref="ref-1",
    )
    card = _minimal_card(
        human_validation_status=HumanValidationStatus.PROVISIONAL,
        human_correlation=correlation,
    )
    assert card.human_correlation is correlation


def _write_run(runs_dir, run_id, level, rows, identities=None):
    run = runs_dir / run_id
    run.mkdir()
    (run / "raw_results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (run / "summary_meta.json").write_text(json.dumps({"level": level}))
    (run / "model_identities.json").write_text(json.dumps(identities or {}))
    return run


def _capability_profile(*, digest, measured, supported_families, schema_version=None):
    return {
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION if schema_version is None else schema_version,
        "probe_protocol_version": "v1",
        "capability_identity": {"model": {"digest": digest}} if digest is not None else {},
        "declared_capabilities": [],
        "supported_families": supported_families,
        "measured_capabilities": measured,
    }


def _fixed_model_row():
    return {
        "display_name": "model:latest",
        "digest": "sha256:abc123",
        "names_seen": ["model:latest"],
        "class": "general",
        "families": ["text"],
        "parameter_size": "7B",
        "quantization_level": "Q4_K_M",
        "architecture_family": "llama",
        "model_size_bytes": 4_294_967_296,
        "size_gb": 4.0,
        "quality_status": "complete",
        "overall_mean_score": 88.5,
        "overall_rank": 1,
        "tie_band": 1,
        "coverage_ratio": 1.0,
        "completion_rate": 1.0,
        "quality_status_reasons": ["complete"],
        "capability_limited": False,
        "capability_unavailable_tasks": [],
        "capability_measured_failure": False,
        "capability_measured_failure_tasks": [],
        "recovery_limited": False,
        "recovery_exhausted_tasks": [],
        "think_ineffective_tasks": [],
        "long_context_profile": {
            "target_ctx": 64000,
            "target_status": "ready",
            "max_verified_ctx": 64773,
            "score": 100.0,
            "coverage": 1.0,
            "target_tps": 12.0,
            "target_prompt_tps": 220.0,
            "depths": [],
        },
    }


_OLD_CARD_KEYS = {
    "model", "digest", "identity", "quality", "limits", "long_context",
    "kv_compatibility", "evidence_warnings",
}


def test_golden_card_keeps_every_pre_stage_3_1_key_and_value():
    row = _fixed_model_row()
    card = build_card(row)
    assert _OLD_CARD_KEYS <= card.keys()
    assert card["model"] == "model:latest"
    assert card["digest"] == "sha256:abc123"
    assert card["identity"]["families"] == ["text"]
    assert card["identity"]["parameter_size"] == "7B"
    assert card["quality"]["overall_mean_score"] == 88.5
    assert card["quality"]["coverage_ratio"] == 1.0
    assert card["limits"]["capability_limited"] is False
    assert card["long_context"]["target_status"] == "ready"
    assert card["long_context"]["max_verified_ctx"] == 64773
    assert card["kv_compatibility"] == {}


def test_golden_card_adds_stage_3_1_fields_without_disturbing_the_rest():
    row = _fixed_model_row()
    row["capability_profile"] = _capability_profile(
        digest="sha256:abc123",
        measured={"text": {"state": "measured_supported"}},
        supported_families=["text"],
    )
    card = build_card(row)
    assert card["schema_version"] == 2
    assert card["human_validation_status"] == HumanValidationStatus.NOT_EVALUATED.value
    assert "human_correlation" not in card


def test_current_capability_schema_alone_does_not_promote_to_canonical_compatible():
    """Reconciliation regression 1: a bound capability profile whose schema
    version matches today's ``CAPABILITY_SCHEMA_VERSION`` is not, by itself,
    proof of scorer compatibility, task-hash compatibility, or the absence
    of known-broken capability routing -- the actual ``EvidenceTrustClass``
    contract (``evidence.py``). With no authoritative upstream
    classification, the card must stay ``unknown_legacy``, never
    ``canonical_compatible``."""
    row = _fixed_model_row()
    row["capability_profile"] = _capability_profile(
        digest="sha256:abc123",
        measured={"text": {"state": "measured_supported"}},
        supported_families=["text"],
    )
    card = build_card(row)
    assert card["evidence_trust_class"] == EvidenceTrustClass.UNKNOWN_LEGACY.value


def test_old_capability_schema_alone_does_not_promote_to_historical_valid():
    """Reconciliation regression 2: an older capability-profile schema
    version is likewise not, by itself, proof of ``historical_valid`` --
    with no authoritative upstream classification, it also stays
    ``unknown_legacy``. Schema staleness is still surfaced -- as a warning,
    not a trust promotion (see the paired assertion below and
    ``test_capability_schema_staleness_is_a_warning_not_a_trust_promotion``)."""
    row = _fixed_model_row()
    row["capability_profile"] = _capability_profile(
        digest="sha256:abc123",
        measured={"text": {"state": "measured_supported"}},
        supported_families=["text"],
        schema_version=CAPABILITY_SCHEMA_VERSION - 1,
    )
    card = build_card(row)
    assert card["evidence_trust_class"] == EvidenceTrustClass.UNKNOWN_LEGACY.value


def test_authoritative_upstream_trust_class_is_projected_unchanged():
    """Reconciliation regression 3: if an explicit, authoritative
    ``EvidenceTrustClass`` already exists upstream (row-level or nested
    under ``capability_profile``), ``ModelCard`` projects it exactly --
    never reclassifies it, even when it disagrees with what schema-version
    alone would otherwise suggest."""
    row = _fixed_model_row()
    row["capability_profile"] = _capability_profile(
        digest="sha256:abc123",
        measured={"text": {"state": "measured_supported"}},
        supported_families=["text"],
    )
    row["capability_profile"]["evidence_trust_class"] = "calibration_only"
    card = build_card(row)
    assert card["evidence_trust_class"] == EvidenceTrustClass.CALIBRATION_ONLY.value


def test_authoritative_upstream_trust_class_on_the_row_itself_is_projected_unchanged():
    row = _fixed_model_row()
    row["evidence_trust_class"] = "known_invalid"
    card = build_card(row)
    assert card["evidence_trust_class"] == EvidenceTrustClass.KNOWN_INVALID.value


def test_malformed_upstream_trust_class_fails_closed_to_the_conservative_default():
    row = _fixed_model_row()
    row["evidence_trust_class"] = "not-a-real-trust-class"
    card = build_card(row)
    assert card["evidence_trust_class"] == EvidenceTrustClass.UNKNOWN_LEGACY.value


def test_evidence_trust_class_is_unknown_legacy_with_no_bound_capability_profile():
    """Reconciliation regression 4: with no upstream trust classification
    and no capability profile at all, the card fails conservatively to
    ``unknown_legacy``."""
    row = _fixed_model_row()
    card = build_card(row)
    assert card["evidence_trust_class"] == EvidenceTrustClass.UNKNOWN_LEGACY.value


def test_capability_schema_staleness_is_a_warning_not_a_trust_promotion():
    """Reconciliation regression 5: schema staleness is surfaced as its own
    warning, decoupled from ``evidence_trust_class`` -- staleness never
    invents a stronger (or different) trust class."""
    row = _fixed_model_row()
    row["capability_profile"] = _capability_profile(
        digest="sha256:abc123",
        measured={"text": {"state": "measured_supported"}},
        supported_families=["text"],
        schema_version=CAPABILITY_SCHEMA_VERSION - 1,
    )
    card = build_card(row)
    assert card["evidence_trust_class"] == EvidenceTrustClass.UNKNOWN_LEGACY.value
    assert any("schema version" in warning for warning in card["evidence_warnings"])


def test_current_schema_profile_produces_no_schema_staleness_warning():
    row = _fixed_model_row()
    row["capability_profile"] = _capability_profile(
        digest="sha256:abc123",
        measured={"text": {"state": "measured_supported"}},
        supported_families=["text"],
    )
    card = build_card(row)
    assert not any("schema version" in warning for warning in card["evidence_warnings"])


def test_model_card_is_zero_authority_no_consumer_reads_it_for_decisions():
    """Reconciliation regression 6: ``model_card``/``ModelCard`` must not be
    imported by any planner/applicability/judge/routing/campaign/runner/
    benchmark-authorization module -- only its own generation module
    (``model_cards.py``) and this test file may reference it."""
    import pathlib

    package_dir = pathlib.Path(__import__("llm_modelbench").__file__).parent
    allowed = {"model_card.py", "model_cards.py"}
    offenders = []
    for path in sorted(package_dir.glob("*.py")):
        if path.name in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        if "model_card" in text.lower() and "model_cards" not in path.name.lower():
            # Only flag genuine references to the model_card module/objects,
            # not incidental substring hits (there are none in this codebase
            # today, but keep the check honest rather than a bare substring).
            if (
                "import model_card" in text
                or "from .model_card import" in text
                or "from llm_modelbench.model_card import" in text
                or "ModelCard(" in text
                or "build_model_card(" in text
            ):
                offenders.append(path.name)
    assert offenders == []


def test_no_rendered_output_implies_human_validation_by_default():
    row = _fixed_model_row()
    card = build_card(row)
    assert card["human_validation_status"] == HumanValidationStatus.NOT_EVALUATED.value
    from llm_modelbench.model_card import render_model_card_markdown
    markdown = render_model_card_markdown(card)
    assert "does not imply human validation" in markdown
    assert "**validated**" not in markdown.lower()


def test_measured_family_evidence_agrees_between_model_card_and_master_summary(tmp_path):
    """Same fixture shape as ``test_rankings.py``'s Stage 3.0A regression
    (declared ``embedding``, measured ``text``+``tools``): both
    ``master_summary.json`` and the ``ModelCard`` built from its row must
    say ``["text", "tools"]``, never the declared-only ``["embedding"]``."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _write_run(
        runs_dir, "r1", "full",
        [
            {"model": "custom-model", "task": "py_anagram", "category": "coding_python",
             "family": "text", "score": 100.0,
             "task_hash": rankings._CURRENT_HASHES["py_anagram"],
             "capabilities_declared": ["embedding"],
             "timestamp": "2026-01-01T00:00:00Z"},
            {"model": "custom-model", "task": "agent_native_tool_call", "category": "agentic_tool",
             "family": "tools", "score": 100.0,
             "task_hash": rankings._CURRENT_HASHES["agent_native_tool_call"],
             "capabilities_declared": ["embedding"],
             "timestamp": "2026-01-01T00:00:00Z"},
        ],
        identities={"custom-model": {"digest": "d-custom"}},
    )
    rankings_dir = tmp_path / "rankings"
    rankings.write_rankings(runs_dir, rankings_dir)
    master_row = json.loads((rankings_dir / "master_summary.json").read_text())[0]
    assert master_row["families"] == ["text", "tools"]

    card = build_card(master_row)
    assert card["identity"]["families"] == ["text", "tools"]
    assert card["identity"]["families"] == master_row["families"]


def test_generate_model_cards_end_to_end_still_writes_json_and_markdown(tmp_path):
    rankings_dir = tmp_path / "rankings"
    rankings_dir.mkdir()
    (rankings_dir / "master_summary.json").write_text(json.dumps([_fixed_model_row()]))
    out = tmp_path / "cards"
    result = generate_model_cards(rankings_dir, out)
    assert result["models"] == 1
    written = json.loads((out / "model_latest.json").read_text())
    assert written["evidence_trust_class"] == EvidenceTrustClass.UNKNOWN_LEGACY.value
    assert written["human_validation_status"] == HumanValidationStatus.NOT_EVALUATED.value
    markdown = (out / "model_latest.md").read_text()
    assert "Evidence trust and human validation" in markdown


def test_help_page_no_longer_falsely_claims_model_cards_drive_routing():
    """Stage 3 architecture proposal §9 acceptance criterion: the
    ``rankings_v31.py:399`` "operating cards for model routing decisions"
    claim must be corrected or made true. No code anywhere reads
    ``model_cards/`` output as a decision input (confirmed by the proposal's
    own source-map audit, §1.3) -- so this is a correction, not a new
    consumer."""
    from llm_modelbench import rankings_v31
    import inspect

    source = inspect.getsource(rankings_v31._help_html)
    assert "routing decisions" not in source
    assert "model_cards/" in source
