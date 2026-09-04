"""Anvil Stage 3B.3E -- the local GGUF artifact resolver.

Pure filesystem behaviour: exact-path resolution + real byte hashing, and
every fail-closed edge (missing / unreadable / directory / ambiguous /
unhashable / nothing configured / no model reference). No subprocess, no
network, no arbitrary-tree walk.
"""
from __future__ import annotations

import hashlib
import os

import pytest

from llm_modelbench.local_artifact_resolver import (
    LocalArtifactStatus,
    resolve_local_gguf_artifact,
)

_BYTES = b"GGUF\x00 fake weight bytes for the resolver test"
_SHA = hashlib.sha256(_BYTES).hexdigest()


@pytest.fixture()
def gguf_file(tmp_path):
    p = tmp_path / "my-model-q4.gguf"
    p.write_bytes(_BYTES)
    return p


# ---------------------------------------------------------------------------
# exact local GGUF path resolves and hashes the real bytes
# ---------------------------------------------------------------------------
def test_exact_map_path_resolves_and_hashes_actual_bytes(gguf_file):
    out = resolve_local_gguf_artifact(
        "my-model", artifacts_map={"my-model": str(gguf_file)}, root_dir=None
    )
    assert out.ok and out.status is LocalArtifactStatus.RESOLVED
    assert out.resolved_path == str(gguf_file)
    assert out.verified_sha256 == _SHA
    assert out.size_bytes == len(_BYTES)
    assert out.source == "map"


def test_root_dir_single_match_resolves_and_hashes(gguf_file):
    out = resolve_local_gguf_artifact(
        "my-model", artifacts_map={}, root_dir=str(gguf_file.parent)
    )
    assert out.ok
    assert out.resolved_path == str(gguf_file)
    assert out.verified_sha256 == _SHA
    assert out.source == "root"


def test_map_takes_precedence_over_root(tmp_path):
    a = tmp_path / "a" / "model.gguf"
    a.parent.mkdir()
    a.write_bytes(b"AAA")
    b_dir = tmp_path / "b"
    b_dir.mkdir()
    (b_dir / "model.gguf").write_bytes(b"BBBBBB")
    out = resolve_local_gguf_artifact(
        "model", artifacts_map={"model": str(a)}, root_dir=str(b_dir)
    )
    assert out.resolved_path == str(a)
    assert out.verified_sha256 == hashlib.sha256(b"AAA").hexdigest()


# ---------------------------------------------------------------------------
# fail-closed edges
# ---------------------------------------------------------------------------
def test_no_model_ref_is_fail_closed_not_error():
    out = resolve_local_gguf_artifact(None, artifacts_map={"x": "/x.gguf"}, root_dir="/d")
    assert not out.ok and out.status is LocalArtifactStatus.NO_MODEL_REF
    assert out.verified_sha256 is None


def test_nothing_configured_is_fail_closed():
    out = resolve_local_gguf_artifact("m", artifacts_map={}, root_dir=None)
    assert out.status is LocalArtifactStatus.NOT_CONFIGURED
    assert not out.ok


def test_missing_map_path_fails_closed(tmp_path):
    out = resolve_local_gguf_artifact(
        "m", artifacts_map={"m": str(tmp_path / "nope.gguf")}, root_dir=None
    )
    assert out.status is LocalArtifactStatus.MISSING
    assert out.verified_sha256 is None


def test_directory_path_fails_closed(tmp_path):
    out = resolve_local_gguf_artifact(
        "m", artifacts_map={"m": str(tmp_path)}, root_dir=None
    )
    assert out.status is LocalArtifactStatus.NOT_A_FILE
    assert out.verified_sha256 is None


def test_unreadable_map_path_fails_closed(tmp_path):
    p = tmp_path / "locked.gguf"
    p.write_bytes(_BYTES)
    os.chmod(p, 0o000)
    try:
        out = resolve_local_gguf_artifact(
            "m", artifacts_map={"m": str(p)}, root_dir=None
        )
    finally:
        os.chmod(p, 0o644)
    if os.geteuid() == 0:
        pytest.skip("root bypasses file mode; unreadable path cannot be simulated")
    assert out.status is LocalArtifactStatus.HASH_FAILED
    assert out.verified_sha256 is None


def test_ambiguous_root_match_fails_closed_and_lists_candidates(tmp_path):
    (tmp_path / "my-model-q4.gguf").write_bytes(b"one")
    (tmp_path / "my-model-q8.gguf").write_bytes(b"two")
    out = resolve_local_gguf_artifact(
        "my-model", artifacts_map={}, root_dir=str(tmp_path)
    )
    assert out.status is LocalArtifactStatus.AMBIGUOUS
    assert not out.ok and out.verified_sha256 is None
    assert len(out.candidate_paths) == 2
    assert all(c.endswith(".gguf") for c in out.candidate_paths)


def test_root_scan_is_not_recursive(tmp_path):
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "my-model.gguf").write_bytes(_BYTES)
    out = resolve_local_gguf_artifact(
        "my-model", artifacts_map={}, root_dir=str(tmp_path)
    )
    assert out.status is LocalArtifactStatus.MISSING  # never descends into sub/


def test_root_scan_ignores_non_gguf_files(tmp_path):
    (tmp_path / "my-model.txt").write_bytes(b"notes")
    (tmp_path / "my-model.safetensors").write_bytes(b"weights")
    out = resolve_local_gguf_artifact(
        "my-model", artifacts_map={}, root_dir=str(tmp_path)
    )
    assert out.status is LocalArtifactStatus.MISSING


def test_root_missing_directory_fails_closed(tmp_path):
    out = resolve_local_gguf_artifact(
        "m", artifacts_map={}, root_dir=str(tmp_path / "no-such-dir")
    )
    assert out.status is LocalArtifactStatus.MISSING


# ---------------------------------------------------------------------------
# a caller-/config-claimed hash cannot override the file bytes
# ---------------------------------------------------------------------------
def test_caller_claimed_hash_cannot_override_file_bytes(gguf_file):
    """The map value is a *path*; there is no hash input the resolver could
    trust. The returned sha is always the hash of the actual bytes. This test
    pins that the API has no hash-claim parameter and the result is bound to
    the file content, so a mutation that 'trusts a supplied hash' has nothing
    to trust and the byte hash is authoritative."""
    out = resolve_local_gguf_artifact(
        "my-model", artifacts_map={"my-model": str(gguf_file)}, root_dir=None
    )
    assert out.verified_sha256 == hashlib.sha256(gguf_file.read_bytes()).hexdigest()
    # rewrite the file: the resolver re-hashes, it never caches a claimed value
    gguf_file.write_bytes(b"different bytes entirely")
    out2 = resolve_local_gguf_artifact(
        "my-model", artifacts_map={"my-model": str(gguf_file)}, root_dir=None
    )
    assert out2.verified_sha256 == hashlib.sha256(b"different bytes entirely").hexdigest()
    assert out2.verified_sha256 != out.verified_sha256


def test_hasher_seam_is_injected_and_receives_the_resolved_path(gguf_file):
    seen = {}

    def _fake_hasher(path):
        seen["path"] = path
        return "sha256:" + "a" * 64

    out = resolve_local_gguf_artifact(
        "my-model",
        artifacts_map={"my-model": str(gguf_file)},
        root_dir=None,
        hasher=_fake_hasher,
    )
    assert str(seen["path"]) == str(gguf_file)
    assert out.verified_sha256 == "a" * 64  # normalised: sha256: prefix stripped


def test_to_dict_is_json_serialisable(gguf_file):
    import json

    out = resolve_local_gguf_artifact(
        "my-model", artifacts_map={"my-model": str(gguf_file)}, root_dir=None
    )
    json.dumps(out.to_dict())
