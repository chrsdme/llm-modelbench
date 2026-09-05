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
_T_ARRAY = 9


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


def _kv_array_u32(key: str, values: list) -> bytes:
    key_b = key.encode("utf-8")
    body = struct.pack("<I", _T_UINT32) + struct.pack("<Q", len(values))
    for v in values:
        body += struct.pack("<I", v)
    return struct.pack("<Q", len(key_b)) + key_b + struct.pack("<I", _T_ARRAY) + body


def _kv_array_string(key: str, values: list) -> bytes:
    key_b = key.encode("utf-8")
    body = struct.pack("<I", _T_STRING) + struct.pack("<Q", len(values))
    for v in values:
        v_b = v.encode("utf-8")
        body += struct.pack("<Q", len(v_b)) + v_b
    return struct.pack("<Q", len(key_b)) + key_b + struct.pack("<I", _T_ARRAY) + body


def _nested_array_bytes(depth: int) -> bytes:
    """A single KV pair whose value is an array-of-array-of-...-of-u32,
    nested ``depth`` levels deep, with exactly one element at each level."""
    key_b = b"adversarial.nested"
    # Innermost: an array of one u32.
    tail = struct.pack("<I", _T_UINT32) + struct.pack("<Q", 1) + struct.pack("<I", 7)
    for _ in range(depth - 1):
        tail = struct.pack("<I", _T_ARRAY) + struct.pack("<Q", 1) + tail
    return struct.pack("<Q", len(key_b)) + key_b + struct.pack("<I", _T_ARRAY) + tail


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


# --------------------------------------------------------------------------
# D. array metadata: benign, adversarial-nested, and large-vocab-shaped
# --------------------------------------------------------------------------
def test_benign_shallow_array_metadata_does_not_block_resolution(tmp_path):
    """A flat array-of-strings and array-of-u32 (shapes real GGUF files use
    for e.g. general.tags / rope.dimension_sections) must not prevent the
    required scalar keys from resolving."""
    pairs = _llama_kv_pairs(kv_heads=8, key_length=128, value_length=128)
    pairs = pairs[:1] + [
        _kv_array_string("general.tags", ["chat", "code", "instruct"]),
        _kv_array_u32("llama.rope.dimension_sections", [16, 16, 16, 16]),
    ] + pairs[1:]
    data = _gguf_bytes(pairs)
    res = resolve_gguf_architecture(_write(tmp_path, data))
    assert res.ok
    assert res.architecture_name == "llama"


def test_deeply_nested_array_fails_closed_as_header_too_large(tmp_path):
    """A crafted array-of-array-of-...-of-u32 nested past the depth ceiling
    must fail closed with HEADER_TOO_LARGE, not raise RecursionError."""
    pairs = [_kv_string("general.architecture", "llama"), _nested_array_bytes(depth=50)]
    data = _gguf_bytes(pairs)
    res = resolve_gguf_architecture(_write(tmp_path, data))
    assert res.status is GGUFMetadataStatus.HEADER_TOO_LARGE
    assert not res.ok


def test_shallow_nested_array_under_ceiling_still_resolves(tmp_path):
    """Nesting under the depth ceiling is legitimate GGUF shape and must not
    be refused -- only nesting past the ceiling fails closed."""
    pairs = _llama_kv_pairs(kv_heads=8, key_length=128, value_length=128)
    pairs = pairs[:1] + [_nested_array_bytes(depth=3)] + pairs[1:]
    data = _gguf_bytes(pairs)
    res = resolve_gguf_architecture(_write(tmp_path, data))
    assert res.ok


def test_large_vocab_shaped_arrays_after_architecture_keys_still_resolve(tmp_path):
    """Real GGUF files place large tokenizer vocabulary arrays (hundreds of
    thousands of short strings) among the metadata KV pairs. Whether those
    arrays are declared before or after the architecture-prefixed scalar
    keys must not affect resolution -- the parser must skip them without
    materialising every element, regardless of key order."""
    arch_pairs = _llama_kv_pairs(kv_heads=8, key_length=128, value_length=128)
    vocab = _kv_array_string("tokenizer.ggml.tokens", [f"tok{i}" for i in range(20_000)])
    # Architecture keys placed AFTER the large array -- order-independent.
    data = _gguf_bytes([vocab] + arch_pairs)
    res = resolve_gguf_architecture(_write(tmp_path, data))
    assert res.ok
    assert res.architecture_name == "llama"
    assert res.architecture["head_dimension"] == 128


def test_large_array_skip_does_not_materialise_elements(tmp_path, monkeypatch):
    """A large array-of-strings must be skipped (seeked past) rather than
    read+decoded element by element -- proven by an instrumented handle
    that counts materialising reads separately from the seek-based skip."""
    arch_pairs = _llama_kv_pairs(kv_heads=8, key_length=128, value_length=128)
    vocab = _kv_array_string("tokenizer.ggml.tokens", [f"token_{i}" for i in range(50_000)])
    data = _gguf_bytes(arch_pairs + [vocab])
    path = _write(tmp_path, data)

    import llm_modelbench.gguf_metadata as gguf_metadata

    real_open = gguf_metadata.Path.open
    counted = {"bytes_read": 0, "seek_calls": 0}

    class _CountingHandle:
        def __init__(self, handle):
            self._handle = handle

        def read(self, n):
            data = self._handle.read(n)
            counted["bytes_read"] += len(data)
            return data

        def seek(self, *args):
            counted["seek_calls"] += 1
            return self._handle.seek(*args)

        def close(self):
            self._handle.close()

    def _counting_open(self, mode="r"):
        return _CountingHandle(real_open(self, mode))

    monkeypatch.setattr(gguf_metadata.Path, "open", _counting_open)
    res = resolve_gguf_architecture(path)
    assert res.ok
    # Each string element's 8-byte length prefix is still read (needed to
    # know how far to skip), but the string *payload* itself -- the bulk of
    # a real vocabulary array's bytes -- is skipped via seek, not decoded.
    # 50,000 elements * 8-byte length prefix is the expected materialised
    # floor; the actual string bytes (~450 KB of "token_N" payloads) must
    # not be read.
    assert counted["bytes_read"] < 50_000 * 8 + 4096
    assert counted["seek_calls"] > 0


def test_invalid_utf8_in_string_value_fails_closed(tmp_path):
    """A declared string length pointing at invalid UTF-8 bytes must fail
    closed as truncated/malformed, not raise UnicodeDecodeError."""
    key_b = b"general.architecture"
    bad_bytes = b"\xff\xfe\x00\x01"
    body = (
        struct.pack("<Q", len(key_b)) + key_b
        + struct.pack("<I", _T_STRING)
        + struct.pack("<Q", len(bad_bytes)) + bad_bytes
    )
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 1)
    res = resolve_gguf_architecture(_write(tmp_path, header + body))
    assert res.status is GGUFMetadataStatus.TRUNCATED_OR_MALFORMED
    assert not res.ok
