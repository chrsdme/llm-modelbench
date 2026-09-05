"""Anvil Stage 3B.4 corrective (third): bounded GGUF header metadata reader.

Builds minimal, spec-conforming (and deliberately broken) GGUF byte streams
in-memory -- no real model file, no llama.cpp, no network -- to exercise the
parser's fail-closed contract precisely.
"""
from __future__ import annotations

import struct

from llm_modelbench.gguf_metadata import (
    GGUFMetadataStatus,
    KV_CACHE_DTYPE_BYTES,
    PARALLEL_SEQUENCES,
    resolve_gguf_architecture,
)

_T_UINT32 = 4
_T_STRING = 8


def _kv_string(key: str, value: str) -> bytes:
    key_b = key.encode("utf-8")
    val_b = value.encode("utf-8")
    return (
        struct.pack("<Q", len(key_b)) + key_b
        + struct.pack("<I", _T_STRING)
        + struct.pack("<Q", len(val_b)) + val_b
    )


def _kv_u32(key: str, value: int) -> bytes:
    key_b = key.encode("utf-8")
    return (
        struct.pack("<Q", len(key_b)) + key_b
        + struct.pack("<I", _T_UINT32)
        + struct.pack("<I", value)
    )


def _gguf_bytes(kv_pairs: list, *, version: int = 3, tensor_count: int = 0) -> bytes:
    body = b"".join(kv_pairs)
    header = (
        b"GGUF"
        + struct.pack("<I", version)
        + struct.pack("<Q", tensor_count)
        + struct.pack("<Q", len(kv_pairs))
    )
    return header + body


def _llama_kv_pairs(*, arch="llama", layers=32, heads=32, kv_heads=None,
                     embedding=4096, key_length=None, value_length=None,
                     context_length=8192):
    pairs = [
        _kv_string("general.architecture", arch),
        _kv_u32(f"{arch}.block_count", layers),
        _kv_u32(f"{arch}.attention.head_count", heads),
        _kv_u32(f"{arch}.embedding_length", embedding),
    ]
    if kv_heads is not None:
        pairs.append(_kv_u32(f"{arch}.attention.head_count_kv", kv_heads))
    if key_length is not None:
        pairs.append(_kv_u32(f"{arch}.attention.key_length", key_length))
    if value_length is not None:
        pairs.append(_kv_u32(f"{arch}.attention.value_length", value_length))
    if context_length is not None:
        pairs.append(_kv_u32(f"{arch}.context_length", context_length))
    return pairs


def _write(tmp_path, data: bytes, name: str = "model.gguf") -> str:
    path = tmp_path / name
    path.write_bytes(data)
    return str(path)


# --------------------------------------------------------------------------
# A. valid fixtures
# --------------------------------------------------------------------------
def test_minimal_valid_gguf_resolves_required_metadata(tmp_path):
    data = _gguf_bytes(_llama_kv_pairs(kv_heads=8, key_length=128, value_length=128))
    res = resolve_gguf_architecture(_write(tmp_path, data))
    assert res.ok
    assert res.architecture_name == "llama"
    assert res.model_max_context == 8192
    assert res.architecture == {
        "layer_count": 32,
        "kv_head_count": 8,
        "head_dimension": 128,
        "kv_dtype_bytes": KV_CACHE_DTYPE_BYTES,
        "parallel_sequences": PARALLEL_SEQUENCES,
    }
    assert res.sources["kv_head_count"] == "metadata"
    assert res.sources["key_length"] == "metadata"


def test_missing_head_count_kv_derives_from_head_count(tmp_path):
    data = _gguf_bytes(_llama_kv_pairs(heads=32, key_length=128, value_length=128))
    res = resolve_gguf_architecture(_write(tmp_path, data))
    assert res.ok
    assert res.architecture["kv_head_count"] == 32
    assert res.sources["kv_head_count"] == "derived_kv_heads_equal_heads"


def test_missing_key_length_derives_from_embedding_over_heads(tmp_path):
    data = _gguf_bytes(_llama_kv_pairs(heads=32, embedding=4096, kv_heads=32))
    res = resolve_gguf_architecture(_write(tmp_path, data))
    assert res.ok
    assert res.architecture["head_dimension"] == 128  # 4096 // 32
    assert res.sources["key_length"] == "derived_embedding_per_head"
    assert res.sources["value_length"] == "derived_value_equals_key"


def test_non_divisible_embedding_fails_closed_rather_than_rounding(tmp_path):
    data = _gguf_bytes(_llama_kv_pairs(heads=33, embedding=4096, kv_heads=33))
    res = resolve_gguf_architecture(_write(tmp_path, data))
    assert res.status is GGUFMetadataStatus.INVALID_METADATA
    assert not res.ok


def test_asymmetric_key_value_length_fails_closed(tmp_path):
    data = _gguf_bytes(_llama_kv_pairs(kv_heads=8, key_length=128, value_length=64))
    res = resolve_gguf_architecture(_write(tmp_path, data))
    assert res.status is GGUFMetadataStatus.KV_ASYMMETRIC_KEY_VALUE_LENGTH_UNSUPPORTED
    assert not res.ok


# --------------------------------------------------------------------------
# B. missing / malformed / unsupported
# --------------------------------------------------------------------------
def test_missing_required_metadata_fails_closed(tmp_path):
    pairs = [_kv_string("general.architecture", "llama")]
    data = _gguf_bytes(pairs)
    res = resolve_gguf_architecture(_write(tmp_path, data))
    assert res.status is GGUFMetadataStatus.REQUIRED_METADATA_MISSING
    assert "block_count" in res.detail


def test_missing_architecture_key_fails_closed(tmp_path):
    data = _gguf_bytes([_kv_u32("llama.block_count", 32)])
    res = resolve_gguf_architecture(_write(tmp_path, data))
    assert res.status is GGUFMetadataStatus.ARCHITECTURE_UNKNOWN


def test_wrong_magic_fails_closed(tmp_path):
    data = b"NOPE" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 0)
    res = resolve_gguf_architecture(_write(tmp_path, data))
    assert res.status is GGUFMetadataStatus.NOT_A_GGUF_FILE


def test_unsupported_version_fails_closed(tmp_path):
    data = b"GGUF" + struct.pack("<I", 99) + struct.pack("<Q", 0) + struct.pack("<Q", 0)
    res = resolve_gguf_architecture(_write(tmp_path, data))
    assert res.status is GGUFMetadataStatus.UNSUPPORTED_VERSION


def test_truncated_file_fails_closed_not_crash(tmp_path):
    data = _gguf_bytes(_llama_kv_pairs(kv_heads=8, key_length=128, value_length=128))
    truncated = data[: len(data) - 10]
    res = resolve_gguf_architecture(_write(tmp_path, truncated))
    assert res.status is GGUFMetadataStatus.TRUNCATED_OR_MALFORMED


def test_declared_string_length_exceeding_file_size_fails_closed(tmp_path):
    # A key claims a huge length the file cannot actually hold.
    bogus_key_len = struct.pack("<Q", 10_000_000)
    body = bogus_key_len + b"x" * 8  # nowhere near 10,000,000 bytes
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 1)
    res = resolve_gguf_architecture(_write(tmp_path, header + body))
    assert res.status is GGUFMetadataStatus.TRUNCATED_OR_MALFORMED


def test_metadata_kv_count_over_ceiling_fails_closed(tmp_path):
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 10_000_000_000)
    res = resolve_gguf_architecture(_write(tmp_path, header))
    assert res.status is GGUFMetadataStatus.HEADER_TOO_LARGE


def test_missing_file_is_unreadable_not_a_crash(tmp_path):
    res = resolve_gguf_architecture(str(tmp_path / "does-not-exist.gguf"))
    assert res.status is GGUFMetadataStatus.UNREADABLE


def test_unsupported_architecture_missing_prefixed_keys_fails_closed(tmp_path):
    # general.architecture names an architecture, but none of the
    # architecture-prefixed keys this reader looks up are present under
    # that exact prefix (they're present under a different one) --
    # confirms exact-key lookup, not a substring/suffix match.
    pairs = [
        _kv_string("general.architecture", "weirdarch"),
        _kv_u32("llama.block_count", 32),
        _kv_u32("llama.attention.head_count", 32),
        _kv_u32("llama.embedding_length", 4096),
    ]
    data = _gguf_bytes(pairs)
    res = resolve_gguf_architecture(_write(tmp_path, data))
    assert res.status is GGUFMetadataStatus.REQUIRED_METADATA_MISSING
    assert res.architecture_name == "weirdarch"


# --------------------------------------------------------------------------
# C. does not read tensor payload
# --------------------------------------------------------------------------
def test_parser_never_reads_past_the_metadata_kv_section(tmp_path, monkeypatch):
    """Append a large tensor-data-shaped tail after valid metadata and prove,
    via an instrumented file handle, that total bytes read stops at the
    metadata KV section rather than reading (or seeking into) the tail."""
    valid = _gguf_bytes(_llama_kv_pairs(kv_heads=8, key_length=128, value_length=128))
    tail = b"\xff" * (2 * 1024 * 1024)  # a large tensor-payload stand-in
    path = _write(tmp_path, valid + tail)

    import llm_modelbench.gguf_metadata as gguf_metadata

    real_open = gguf_metadata.Path.open
    counted = {"bytes_read": 0}

    class _CountingHandle:
        def __init__(self, handle):
            self._handle = handle

        def read(self, n):
            data = self._handle.read(n)
            counted["bytes_read"] += len(data)
            return data

        def close(self):
            self._handle.close()

    def _counting_open(self, mode="r"):
        return _CountingHandle(real_open(self, mode))

    monkeypatch.setattr(gguf_metadata.Path, "open", _counting_open)
    res = resolve_gguf_architecture(path)
    assert res.ok
    assert counted["bytes_read"] == len(valid)
    assert counted["bytes_read"] < len(valid) + len(tail)
