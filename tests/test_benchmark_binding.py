"""Anvil Stage 3.2D-2 -- per-run protocol/runtime/model binding + evidence."""
import dataclasses
import json

import pytest

from llm_modelbench.benchmark_binding import (
    build_model_binding, binding_to_dict, protocol_to_dict, row_binding_reference,
)
from llm_modelbench.benchmark_protocol import BenchmarkProtocolError, BenchmarkRuntimeBinding
from llm_modelbench.config import Config
from llm_modelbench.identity import ModelArtifactIdentity
from llm_modelbench.tasks import Task


def _cfg(**kw) -> Config:
    return dataclasses.replace(Config(), **kw)


def _art(set_id="a1") -> ModelArtifactIdentity:
    return ModelArtifactIdentity(artifact_set_id=set_id, primary_sha256=set_id)


def _task(id="t", scorer="exact") -> Task:
    return Task(id=id, category="demo", family="text", scorer=scorer, prompt="hi", meta={"expected": "hi"})


def _build(**over):
    kw = dict(model_artifact_identity=_art(), selected_tasks=[_task("a"), _task("b", "python")],
              cfg=_cfg(), backend="llama_cpp", backend_version="b4000")
    kw.update(over)
    return build_model_binding(**kw)


# --- the binding itself -----------------------------------------------------

def test_same_model_protocol_and_profile_yield_the_same_binding():
    _, a = _build()
    _, b = _build()
    assert a.binding_key() == b.binding_key()
    assert isinstance(a, BenchmarkRuntimeBinding)


def test_model_artifact_change_changes_the_binding():
    _, base = _build()
    _, other = _build(model_artifact_identity=_art("a2"))
    assert base.binding_key() != other.binding_key()


def test_protocol_change_changes_the_binding():
    _, base = _build(selected_tasks=[_task("a")])
    _, changed = _build(selected_tasks=[_task("a"), _task("b")])
    assert base.benchmark_protocol_identity_key != changed.benchmark_protocol_identity_key
    assert base.binding_key() != changed.binding_key()


def test_runtime_profile_config_change_changes_the_binding():
    _, base = _build(cfg=_cfg())
    _, ctx = _build(cfg=_cfg(ctx_override=8192))
    assert base.runtime_profile_identity_key != ctx.runtime_profile_identity_key
    assert base.binding_key() != ctx.binding_key()


def test_backend_version_change_changes_the_binding():
    _, base = _build(backend_version="b4000")
    _, newer = _build(backend_version="b4100")
    assert base.binding_key() != newer.binding_key()


# --- allowed adaptations ---------------------------------------------------

def test_context_override_is_recorded_as_a_used_adaptation():
    _, adapted = _build(cfg=_cfg(ctx_override=16384))
    assert adapted.allowed_adaptations_used == ("context_size",)


def test_plain_run_uses_no_adaptations():
    _, plain = _build(cfg=_cfg())
    assert plain.allowed_adaptations_used == ()


def test_ram_spill_is_never_a_protocol_adaptation():
    # The binding builder never offers `ram_spill`; even if a future edit did,
    # bind_runtime_to_protocol would reject it against ("context_size",).
    from llm_modelbench.benchmark_policy import build_benchmark_protocol
    from llm_modelbench.benchmark_protocol import bind_runtime_to_protocol
    protocol = build_benchmark_protocol([_task("a")], _cfg())
    with pytest.raises(BenchmarkProtocolError, match="not permitted by protocol"):
        bind_runtime_to_protocol(protocol, model_artifact_identity=_art(),
                                 runtime_profile_identity_key="k",
                                 allowed_adaptations_used=("ram_spill",))


# --- spill permission does not fork the binding recipe --------------------

def test_spill_permission_alone_does_not_change_the_binding():
    # allow_ram_spill is an execution-time operator permission; it is
    # identity-bearing in RuntimeIdentity.identity_hash, not in the reusable
    # runtime recipe (D-1 / owner section 9).
    hot = _cfg(); hot.allow_ram_spill = True          # set dynamically by cmd_run, not a Config field
    cold = _cfg(); cold.allow_ram_spill = False
    _, permitted = _build(cfg=hot)
    _, forbidden = _build(cfg=cold)
    assert permitted.binding_key() == forbidden.binding_key()


# --- serialization -------------------------------------------------------

def test_serialized_forms_are_reference_oriented_and_json_safe():
    protocol, binding = _build()
    bd = binding_to_dict(binding)
    pd = protocol_to_dict(protocol)
    rr = row_binding_reference(binding, protocol)
    json.dumps({"b": bd, "p": pd, "r": rr}, allow_nan=False)  # must not raise
    assert bd["benchmark_protocol_identity_key"] == pd["identity_key"]
    assert rr["benchmark_binding_key"] == binding.binding_key()
    assert rr["benchmark_protocol_id"] == "llm-modelbench-core"
    # reference only -- no embedded object
    assert set(bd) == {"binding_key", "model_artifact_set_id", "benchmark_protocol_identity_key",
                       "runtime_profile_identity_key", "allowed_adaptations_used", "provenance"}


def test_row_reference_is_small_and_additive():
    protocol, binding = _build()
    rr = row_binding_reference(binding, protocol)
    assert set(rr) == {"benchmark_binding_key", "benchmark_protocol_identity_key",
                       "benchmark_protocol_id", "benchmark_protocol_version"}


# --- evidence integration (runner) --------------------------------------

def test_mock_run_writes_a_binding_artifact_and_row_references(tmp_path):
    import sys
    sys.path.insert(0, "tests")
    from test_anvil_stage0_characterization_freeze import _run_mock_benchmark

    run_dir = _run_mock_benchmark(tmp_path / "runs", "x")
    artifact = json.loads((run_dir / "benchmark_bindings.json").read_text())
    assert artifact["schema_version"] == 1
    assert artifact["bindings"], "at least one model binding recorded"
    for model, entry in artifact["bindings"].items():
        assert entry["protocol"]["protocol_id"] == "llm-modelbench-core"
        assert entry["binding"]["binding_key"]
        # placement qualifiers are NOT in the binding (amendment section 15.2.1)
        assert "ram_spill" not in json.dumps(entry["binding"])
        assert "full_gpu" not in json.dumps(entry["binding"])

    rows = [json.loads(line) for line in (run_dir / "raw_results.jsonl").read_text().splitlines() if line.strip()]
    referenced = {r["model"] for r in rows if "benchmark_binding_key" in r}
    assert referenced == set(artifact["bindings"])
    # each row's reference matches its model's artifact binding exactly
    by_model = {m: e["row_reference"] for m, e in artifact["bindings"].items()}
    for r in rows:
        if "benchmark_binding_key" in r:
            assert r["benchmark_binding_key"] == by_model[r["model"]]["benchmark_binding_key"]


def test_unmapped_scorer_fails_closed_as_an_operator_refusal():
    # AGENTS.md section 4 rule 3: an unknown scorer must fail closed at
    # canonical protocol construction. build_model_binding is the only live
    # entry point (runner.run, before any row is written). The failure must
    # reach the operator as `run refused: ...`, not a traceback -- which it
    # does because BenchmarkPolicyError subclasses ValueError and cli.py's
    # `cmd_run` wraps runner.run in `except ValueError -> SystemExit("run
    # refused: ...")`.
    from llm_modelbench.benchmark_policy import BenchmarkPolicyError

    assert issubclass(BenchmarkPolicyError, ValueError)
    with pytest.raises(BenchmarkPolicyError, match="no explicit contract version"):
        build_model_binding(
            model_artifact_identity=_art(),
            selected_tasks=[_task("nope", scorer="totally_unmapped_scorer")],
            cfg=_cfg(), backend="llama_cpp", backend_version="b4000",
        )


def test_legacy_rows_without_a_binding_reference_remain_readable(tmp_path):
    # A row dict lacking the binding keys is structurally valid -- the runner
    # adds them under an `if model in benchmark_bindings` guard, exactly like
    # row_identity_reference. Nothing downstream may hard-require them.
    legacy_row = {"model": "old", "task": "t", "score": 100.0}
    assert "benchmark_binding_key" not in legacy_row  # and that's fine
    json.dumps(legacy_row)  # still serializes


def test_placement_evidence_stays_separately_visible(tmp_path):
    import sys
    sys.path.insert(0, "tests")
    from test_anvil_stage0_characterization_freeze import _run_mock_benchmark

    run_dir = _run_mock_benchmark(tmp_path / "runs", "x")
    # runtime_identity.json still carries the concrete execution identity,
    # independent of the binding artifact.
    ri = json.loads((run_dir / "runtime_identity.json").read_text())
    assert ri["identities"], "concrete runtime identity still recorded per model"
    assert (run_dir / "benchmark_bindings.json").exists()
    assert (run_dir / "runtime_identity.json").exists()  # both, side by side


# --- Anvil Stage 3.2E: resume-divergent binding persistence -----------------

def _model_binding_entry(**over):
    """Build a real ``benchmark_bindings[model]`` entry the way runner.run does."""
    from llm_modelbench.benchmark_binding import (
        binding_to_dict, protocol_to_dict, row_binding_reference,
    )
    protocol, binding = _build(**over)
    return {
        "binding": binding_to_dict(binding),
        "protocol": protocol_to_dict(protocol),
        "row_reference": row_binding_reference(binding, protocol),
    }


def test_fresh_run_writes_the_immutable_model_keyed_artifact(tmp_path):
    from llm_modelbench.runner import _persist_benchmark_bindings

    bindings = {"m1": _model_binding_entry()}
    _persist_benchmark_bindings(tmp_path, bindings, resume=False)
    art = json.loads((tmp_path / "benchmark_bindings.json").read_text())
    assert art["schema_version"] == 1
    assert set(art["bindings"]) == {"m1"}
    assert "resume_divergent_bindings" not in art
    # A fresh (non-resume) write is unconditional even if a file is already there.
    before = (tmp_path / "benchmark_bindings.json").read_text()
    _persist_benchmark_bindings(tmp_path, bindings, resume=False)
    assert (tmp_path / "benchmark_bindings.json").read_text() == before


def test_resume_with_identical_binding_does_not_rewrite_the_artifact(tmp_path):
    from llm_modelbench.runner import _persist_benchmark_bindings

    bindings = {"m1": _model_binding_entry()}
    _persist_benchmark_bindings(tmp_path, bindings, resume=False)
    before = (tmp_path / "benchmark_bindings.json").read_text()
    # Same protocol + runtime + artifact on resume -> same binding_key -> no-op.
    _persist_benchmark_bindings(tmp_path, bindings, resume=True)
    assert (tmp_path / "benchmark_bindings.json").read_text() == before


def test_resume_with_a_divergent_binding_appends_it_so_every_row_key_resolves(tmp_path):
    from llm_modelbench.runner import _persist_benchmark_bindings

    original = {"m1": _model_binding_entry()}
    _persist_benchmark_bindings(tmp_path, original, resume=False)
    original_key = original["m1"]["binding"]["binding_key"]

    # A resumed run under a drifted protocol (e.g. --seed / --temperature /
    # task-selection change -- none gated by the runtime-identity resume check).
    drifted = {"m1": _model_binding_entry(cfg=_cfg(seed=99, temperature=0.7))}
    drifted_key = drifted["m1"]["binding"]["binding_key"]
    assert drifted_key != original_key

    _persist_benchmark_bindings(tmp_path, drifted, resume=True)
    art = json.loads((tmp_path / "benchmark_bindings.json").read_text())

    # The immutable model-keyed map is untouched...
    assert art["bindings"]["m1"]["binding"]["binding_key"] == original_key
    # ...and the divergent binding is recorded additively.
    div = art["resume_divergent_bindings"]
    assert [e["binding"]["binding_key"] for e in div] == [drifted_key]
    assert div[0]["model"] == "m1"

    # Every binding_key any row could carry (B1 from the first pass, B2 from the
    # resumed pass) now resolves against the artifact -- Anvil amendment section 19.
    resolvable = {art["bindings"]["m1"]["binding"]["binding_key"]}
    resolvable |= {e["binding"]["binding_key"] for e in div}
    assert {original_key, drifted_key} <= resolvable

    # A second resume with the same drift is idempotent.
    _persist_benchmark_bindings(tmp_path, drifted, resume=True)
    art2 = json.loads((tmp_path / "benchmark_bindings.json").read_text())
    assert [e["binding"]["binding_key"] for e in art2["resume_divergent_bindings"]] == [drifted_key]


def test_unreadable_prior_artifact_on_resume_is_replaced_not_crashed(tmp_path):
    from llm_modelbench.runner import _persist_benchmark_bindings

    (tmp_path / "benchmark_bindings.json").write_text("{ not json")
    bindings = {"m1": _model_binding_entry()}
    _persist_benchmark_bindings(tmp_path, bindings, resume=True)
    art = json.loads((tmp_path / "benchmark_bindings.json").read_text())
    assert set(art["bindings"]) == {"m1"}
