"""Anvil Stage 0 — tests for identity primitives, the evidence ledger,
generic provenance / EffectiveEvidenceResolver, the projection store, and
the locking/atomicity primitives they're built on.

Fully offline: no GPU, no live Ollama/llama.cpp endpoint, filesystem only.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from llm_modelbench.evidence import (
    EffectiveEvidenceResolutionError,
    EffectiveEvidenceResolver,
    EvalResult,
    EvalStatus,
    EvidenceLedger,
    EvidenceLedgerError,
    EvidenceTrustClass,
    FileLock,
    LockHeldError,
    ProjectionStore,
    ProvenanceLink,
    ProvenanceRelation,
    StaleLockError,
    atomic_write_json,
    atomic_write_text,
    new_event_id,
)
from llm_modelbench.identity import (
    ModelArtifactIdentity,
    RuntimeInstanceIdentity,
    RuntimeProfileIdentity,
)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_model_artifact_identity_from_ollama_tag_row_uses_digest():
    row = {"name": "qwen2.5-coder:14b", "digest": "sha256:abc123", "size": 9_000_000_000}
    ident = ModelArtifactIdentity.from_ollama_tag_row(row)
    assert ident.artifact_set_id == "sha256:abc123"
    assert ident.primary_sha256 == "sha256:abc123"
    assert ident.format == "ollama-blob"
    assert ident.size_bytes == 9_000_000_000


def test_model_artifact_identity_from_ollama_tag_row_falls_back_to_name_hash_without_digest():
    row = {"name": "some-model:latest"}
    ident = ModelArtifactIdentity.from_ollama_tag_row(row)
    assert ident.primary_sha256 is None
    assert ident.artifact_set_id  # non-empty, derived deterministically
    # Same input -> same fallback id.
    assert ModelArtifactIdentity.from_ollama_tag_row(row).artifact_set_id == ident.artifact_set_id


def test_model_artifact_identity_from_gguf_path_is_content_addressed_and_stable():
    a1 = ModelArtifactIdentity.from_gguf_path("deadbeef" * 8)
    a2 = ModelArtifactIdentity.from_gguf_path("deadbeef" * 8)
    assert a1.artifact_set_id == a2.artifact_set_id
    assert a1.primary_sha256 == "deadbeef" * 8


def test_model_artifact_identity_gguf_with_mmproj_differs_from_without():
    without_mmproj = ModelArtifactIdentity.from_gguf_path("deadbeef" * 8)
    with_mmproj = ModelArtifactIdentity.from_gguf_path("deadbeef" * 8, mmproj_sha256="cafebabe" * 8)
    assert without_mmproj.artifact_set_id != with_mmproj.artifact_set_id
    assert with_mmproj.auxiliary_artifact_hashes == ("cafebabe" * 8,)


def test_runtime_profile_identity_stable_key_deterministic_and_order_independent_for_flags():
    p1 = RuntimeProfileIdentity(backend="ollama", backend_version="0.5", feature_flags=("a", "b"))
    p2 = RuntimeProfileIdentity(backend="ollama", backend_version="0.5", feature_flags=("b", "a"))
    assert p1.stable_key() == p2.stable_key()


def test_runtime_profile_identity_stable_key_distinguishes_real_differences():
    p1 = RuntimeProfileIdentity(backend="ollama", backend_version="0.5")
    p2 = RuntimeProfileIdentity(backend="ollama", backend_version="0.6")
    p3 = RuntimeProfileIdentity(backend="llama.cpp", backend_version="0.5")
    assert len({p1.stable_key(), p2.stable_key(), p3.stable_key()}) == 3


def test_runtime_instance_identity_key_differs_by_process_but_profile_key_does_not():
    profile = RuntimeProfileIdentity(backend="llama.cpp", template_hash="th1")
    instance_a = RuntimeInstanceIdentity(profile=profile, endpoint="http://x:8080", process_id=4127)
    instance_b = RuntimeInstanceIdentity(profile=profile, endpoint="http://x:8080", process_id=8192)
    # Restarting the same validated profile (PID changes) must not change
    # the profile-level key that capability evidence is bound to.
    assert instance_a.profile.stable_key() == instance_b.profile.stable_key()
    # But it must change the instance-level key that execution evidence uses.
    assert instance_a.instance_key() != instance_b.instance_key()


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------


def test_eval_result_measured_requires_score():
    with pytest.raises(ValueError):
        EvalResult(status=EvalStatus.MEASURED, score=None)


def test_eval_result_non_measured_statuses_allow_null_score():
    result = EvalResult(status=EvalStatus.ENVIRONMENT_SKIPPED, reason="kv_cache_exceeds_vram_budget")
    assert result.score is None
    assert result.status is EvalStatus.ENVIRONMENT_SKIPPED


# ---------------------------------------------------------------------------
# Locking / atomicity
# ---------------------------------------------------------------------------


def test_atomic_write_text_leaves_no_partial_file_and_no_tmp_litter(tmp_path: Path):
    target = tmp_path / "out.txt"
    atomic_write_text(target, "hello")
    assert target.read_text() == "hello"
    leftovers = [p for p in tmp_path.iterdir() if p != target]
    assert leftovers == []


def test_new_event_id_is_deterministic_for_identical_input():
    a = new_event_id("run_row", {"task": "x", "score": 1})
    b = new_event_id("run_row", {"task": "x", "score": 1})
    assert a == b


def test_new_event_id_differs_for_different_payload():
    a = new_event_id("run_row", {"task": "x", "score": 1})
    b = new_event_id("run_row", {"task": "x", "score": 2})
    assert a != b


def test_file_lock_round_trip(tmp_path: Path):
    lock = FileLock(tmp_path / "l.lock")
    with lock:
        assert lock.path.exists()
    assert not lock.path.exists()


def test_file_lock_blocks_second_holder_non_blocking(tmp_path: Path):
    path = tmp_path / "l.lock"
    first = FileLock(path)
    first.acquire()
    try:
        second = FileLock(path)
        with pytest.raises(LockHeldError):
            second.acquire(blocking=False)
    finally:
        first.release()


def test_file_lock_reclaims_stale_lock_from_dead_pid(tmp_path: Path):
    path = tmp_path / "l.lock"
    # Simulate a lock file left behind by a process that no longer exists.
    # PID 1 always exists on a real system; use a PID very unlikely to be
    # alive instead -- start a short-lived child, wait for it to exit, then
    # use its now-dead PID.
    import subprocess

    proc = subprocess.Popen(["true"])
    dead_pid = proc.pid
    proc.wait()

    import socket

    atomic_write_json(
        path,
        {"pid": dead_pid, "hostname": socket.gethostname(), "acquired_at": time.time(), "phase": "stale"},
    )
    lock = FileLock(path)
    # Must reclaim rather than raise LockHeldError or hang.
    lock.acquire(blocking=False)
    lock.release()


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root bypasses directory permission checks, so unlink() would succeed anyway",
)
def test_file_lock_raises_stale_lock_error_when_reclaim_itself_fails(tmp_path: Path):
    """Owner is unambiguously dead, but clearing the stale lock file fails
    (e.g. a permissions problem) -- must surface as StaleLockError, not
    silently loop forever or misreport as LockHeldError (which would imply
    a live owner)."""
    import socket
    import subprocess

    proc = subprocess.Popen(["true"])
    dead_pid = proc.pid
    proc.wait()

    lock_dir = tmp_path / "locked_dir"
    lock_dir.mkdir()
    path = lock_dir / "l.lock"
    atomic_write_json(
        path,
        {"pid": dead_pid, "hostname": socket.gethostname(), "acquired_at": time.time(), "phase": "stale"},
    )
    lock_dir.chmod(0o555)  # read+execute only: unlink() inside it will fail
    try:
        lock = FileLock(path)
        with pytest.raises(StaleLockError):
            lock.acquire(blocking=False)
    finally:
        lock_dir.chmod(0o755)  # restore so tmp_path cleanup can remove it


def test_file_lock_never_assumes_a_different_hosts_lock_is_stale(tmp_path: Path):
    path = tmp_path / "l.lock"
    atomic_write_json(
        path,
        {"pid": 999999, "hostname": "some-other-machine", "acquired_at": time.time(), "phase": "x"},
    )
    lock = FileLock(path)
    with pytest.raises(LockHeldError):
        lock.acquire(blocking=False)
    assert path.exists()  # must not have been deleted


# ---------------------------------------------------------------------------
# EvidenceLedger
# ---------------------------------------------------------------------------


def test_ledger_append_and_get(tmp_path: Path):
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    record = ledger.append("primary_row", {"task": "py_anagram", "score": 100.0})
    fetched = ledger.get(record.record_id)
    assert fetched is not None
    assert fetched.payload["task"] == "py_anagram"


def test_ledger_append_is_idempotent_for_identical_event(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"
    ledger = EvidenceLedger(path)
    r1 = ledger.append("primary_row", {"task": "x", "score": 1})
    r2 = ledger.append("primary_row", {"task": "x", "score": 1})
    assert r1.record_id == r2.record_id
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    assert len(lines) == 1  # not appended twice


def test_ledger_append_rejects_id_collision_with_different_content(tmp_path: Path):
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    ledger.append("primary_row", {"task": "x", "score": 1}, record_id="fixed-id")
    with pytest.raises(EvidenceLedgerError):
        ledger.append("primary_row", {"task": "x", "score": 2}, record_id="fixed-id")


def test_ledger_never_mutates_or_deletes_no_such_method_exists(tmp_path: Path):
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    assert not hasattr(ledger, "update")
    assert not hasattr(ledger, "delete")
    assert not hasattr(ledger, "remove")


def test_ledger_persists_across_instances(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"
    ledger1 = EvidenceLedger(path)
    record = ledger1.append("primary_row", {"task": "x", "score": 1})
    ledger2 = EvidenceLedger(path)
    assert ledger2.get(record.record_id) is not None


def test_ledger_malformed_json_line_is_skipped_not_fatal(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"
    ledger = EvidenceLedger(path)
    good = ledger.append("primary_row", {"task": "a"})
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not valid json\n")
    fresh = EvidenceLedger(path)
    assert fresh.get(good.record_id) is not None
    assert list(fresh.all()) == [good]
    reasons = fresh.malformed_lines()
    assert len(reasons) == 1
    line_number, reason = reasons[0]
    assert line_number == 2
    assert "JSONDecodeError" in reason


def test_ledger_malformed_missing_required_key_is_skipped_not_fatal(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"
    ledger = EvidenceLedger(path)
    good = ledger.append("primary_row", {"task": "a"})
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"record_id": "x", "record_type": "primary_row"}) + "\n")
    fresh = EvidenceLedger(path)
    assert [r.record_id for r in fresh.all()] == [good.record_id]
    reasons = fresh.malformed_lines()
    assert len(reasons) == 1
    assert "KeyError" in reasons[0][1]


def test_ledger_malformed_invalid_enum_value_is_skipped_not_fatal(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"
    ledger = EvidenceLedger(path)
    good = ledger.append("primary_row", {"task": "a"})
    bad_line = json.dumps(
        {
            "record_id": "y",
            "record_type": "primary_row",
            "payload": {},
            "provenance": [],
            "trust_class": "not_a_real_trust_class",
        }
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(bad_line + "\n")
    fresh = EvidenceLedger(path)
    assert [r.record_id for r in fresh.all()] == [good.record_id]
    reasons = fresh.malformed_lines()
    assert len(reasons) == 1
    assert "ValueError" in reasons[0][1]


def test_ledger_no_malformed_lines_on_a_clean_ledger(tmp_path: Path):
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    ledger.append("primary_row", {"task": "a"})
    assert ledger.malformed_lines() == ()


def test_ledger_find_by_record_type(tmp_path: Path):
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    ledger.append("primary_row", {"task": "a"})
    ledger.append("primary_row", {"task": "b"})
    ledger.append("judge_result", {"task": "a", "score": 90})
    assert len(ledger.find(record_type="primary_row")) == 2
    assert len(ledger.find(record_type="judge_result")) == 1


def test_ledger_trust_class_defaults_and_round_trips(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"
    ledger = EvidenceLedger(path)
    record = ledger.append(
        "primary_row", {"task": "x"}, trust_class=EvidenceTrustClass.CALIBRATION_ONLY
    )
    reloaded = EvidenceLedger(path).get(record.record_id)
    assert reloaded is not None
    assert reloaded.trust_class is EvidenceTrustClass.CALIBRATION_ONLY


def test_legacy_ledger_line_without_trust_class_key_deserialises_unchanged(tmp_path: Path):
    """Stage 3B.2 must NOT touch the historical LedgerRecord.from_dict
    fallback (owner rule: 'do not change deserialization defaults in a way
    that retrospectively upgrades old records' -- but equally must not
    break them). A pre-trust-class line still loads with the unchanged
    historical default."""
    path = tmp_path / "ledger.jsonl"
    legacy_line = json.dumps(
        {
            "record_id": "legacy-1",
            "record_type": "primary_row",
            "payload": {"task": "old"},
            "provenance": [],
        }
    )
    path.write_text(legacy_line + "\n", encoding="utf-8")
    fresh = EvidenceLedger(path)
    assert fresh.malformed_lines() == ()
    rec = fresh.get("legacy-1")
    assert rec is not None
    assert rec.trust_class is EvidenceTrustClass.CANONICAL_COMPATIBLE


# ---------------------------------------------------------------------------
# EffectiveEvidenceResolver
# ---------------------------------------------------------------------------


def test_resolver_returns_the_record_itself_when_never_superseded(tmp_path: Path):
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    primary = ledger.append("primary_row", {"task": "x", "score": 100})
    resolver = EffectiveEvidenceResolver(ledger)
    assert resolver.resolve(primary.record_id).record_id == primary.record_id


def test_resolver_follows_a_supersession_chain_to_the_terminal(tmp_path: Path):
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    primary = ledger.append("primary_row", {"task": "x", "score": 0, "reason": "empty_output"})
    recovery = ledger.append(
        "recovery_row",
        {"task": "x", "score": 80},
        provenance=[ProvenanceLink(ProvenanceRelation.SUPERSEDES, primary.record_id)],
    )
    correction = ledger.append(
        "corrective_row",
        {"task": "x", "score": 85},
        provenance=[ProvenanceLink(ProvenanceRelation.SUPERSEDES, recovery.record_id)],
    )
    resolver = EffectiveEvidenceResolver(ledger)
    effective = resolver.resolve(primary.record_id)
    assert effective.record_id == correction.record_id
    # Historical evidence is untouched -- the primary and recovery records
    # are still individually fetchable.
    assert ledger.get(primary.record_id) is not None
    assert ledger.get(recovery.record_id) is not None


def test_resolver_detects_a_cycle(tmp_path: Path):
    # A record's record_id is content-addressed from (type, payload), so a
    # genuine cycle needs distinct ids that point back to one another:
    # ra -> (supersedes) rc -> (supersedes) rb -> (supersedes) ra.
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    ledger.append("row", {"n": "a"}, record_id="ra",
                   provenance=[ProvenanceLink(ProvenanceRelation.SUPERSEDES, "rc")])
    ledger.append("row", {"n": "b"}, record_id="rb",
                   provenance=[ProvenanceLink(ProvenanceRelation.SUPERSEDES, "ra")])
    ledger.append("row", {"n": "c"}, record_id="rc",
                   provenance=[ProvenanceLink(ProvenanceRelation.SUPERSEDES, "rb")])
    resolver = EffectiveEvidenceResolver(ledger)
    with pytest.raises(EffectiveEvidenceResolutionError, match="cycle"):
        resolver.resolve("ra")


def test_resolver_detects_ambiguous_fork(tmp_path: Path):
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    primary = ledger.append("row", {"n": "primary"}, record_id="p")
    ledger.append("row", {"n": "fork1"}, record_id="f1",
                   provenance=[ProvenanceLink(ProvenanceRelation.SUPERSEDES, "p")])
    ledger.append("row", {"n": "fork2"}, record_id="f2",
                   provenance=[ProvenanceLink(ProvenanceRelation.SUPERSEDES, "p")])
    resolver = EffectiveEvidenceResolver(ledger)
    with pytest.raises(EffectiveEvidenceResolutionError, match="ambiguous"):
        resolver.resolve(primary.record_id)


def test_resolver_detects_missing_source(tmp_path: Path):
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    ledger.append("row", {"n": "orphan"}, record_id="orphan",
                   provenance=[ProvenanceLink(ProvenanceRelation.SUPERSEDES, "does-not-exist")])
    resolver = EffectiveEvidenceResolver(ledger)
    with pytest.raises(EffectiveEvidenceResolutionError, match="missing source"):
        resolver.resolve("does-not-exist")


def test_resolver_unknown_record_id_raises(tmp_path: Path):
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    resolver = EffectiveEvidenceResolver(ledger)
    with pytest.raises(EffectiveEvidenceResolutionError):
        resolver.resolve("nope")


# ---------------------------------------------------------------------------
# ProjectionStore
# ---------------------------------------------------------------------------


def test_projection_store_write_then_set_current_round_trips(tmp_path: Path):
    store = ProjectionStore(tmp_path / "projections")
    assert store.get_current("rankings") is None
    v1 = store.write_version("rankings", {"models": ["a"]})
    store.set_current("rankings", v1)
    assert store.get_current("rankings") == {"models": ["a"]}


def test_projection_store_rollback_is_cheap_and_exact(tmp_path: Path):
    store = ProjectionStore(tmp_path / "projections")
    v1 = store.write_version("rankings", {"models": ["a"]})
    store.set_current("rankings", v1)
    v2 = store.write_version("rankings", {"models": ["a", "b"]})
    store.set_current("rankings", v2)
    assert store.get_current("rankings") == {"models": ["a", "b"]}

    # A "bad adoption" recovers by moving the pointer back -- not a rebuild.
    store.rollback("rankings", v1)
    assert store.get_current("rankings") == {"models": ["a"]}
    # v2 is not deleted -- it's just no longer current.
    assert store.get_version("rankings", v2) == {"models": ["a", "b"]}


def test_projection_store_set_current_rejects_unknown_version(tmp_path: Path):
    store = ProjectionStore(tmp_path / "projections")
    with pytest.raises(FileNotFoundError):
        store.set_current("rankings", "no-such-version")


def test_projection_store_writing_a_version_does_not_move_current(tmp_path: Path):
    store = ProjectionStore(tmp_path / "projections")
    v1 = store.write_version("rankings", {"models": ["a"]})
    store.set_current("rankings", v1)
    store.write_version("rankings", {"models": ["a", "b"]})  # not set current
    assert store.get_current("rankings") == {"models": ["a"]}


def test_projection_store_independent_projection_ids_do_not_interfere(tmp_path: Path):
    store = ProjectionStore(tmp_path / "projections")
    v1 = store.write_version("rankings", {"kind": "rankings"})
    store.set_current("rankings", v1)
    v2 = store.write_version("model_cards", {"kind": "cards"})
    store.set_current("model_cards", v2)
    assert store.get_current("rankings") == {"kind": "rankings"}
    assert store.get_current("model_cards") == {"kind": "cards"}
