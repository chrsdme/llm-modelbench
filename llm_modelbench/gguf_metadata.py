"""Anvil Stage 3B.4 corrective (third) -- bounded, read-only GGUF header
metadata extraction for the pre-spawn managed KV-cache estimate.

This module answers exactly one question: given a local ``.gguf`` file
already resolved and content-hashed by
:mod:`~llm_modelbench.local_artifact_resolver`, what does its own header
declare about the model architecture needed to derive a KV-cache size for a
requested context, *before* any process is spawned and without loading the
model?

**Boundaries (deliberate, not accidental):**

* Reads only the GGUF magic/version/counts and the metadata key-value
  section -- never tensor data, never tensor names, never a full-file
  scan. Every value is checked against declared sizes and hard ceilings
  before it is *materialised* into a Python object
  (:data:`_MAX_MATERIALISED_BYTES`); array/string payloads not needed for
  the KV estimate are skipped by seeking past their declared span rather
  than being read and decoded, bounded by a separate, more generous span
  ceiling (:data:`_MAX_SPAN_BYTES`) and a small hard cap on array nesting
  depth (:data:`_MAX_ARRAY_DEPTH`) -- so a real file's multi-hundred-
  thousand-entry tokenizer vocabulary does not itself exhaust the
  materialisation budget, while a hostile deeply-nested or oversized
  declared length still cannot force unbounded work. This bounds the
  *work* the parser will do on a forged header; it does not itself prove
  the file's tensor data actually begins where a forged metadata count or
  length would place it. A valid GGUF file's metadata declarations do stop
  before the tensor metadata/data sections by construction, so a
  well-formed file is read correctly. A hostile file with forged
  counts/lengths is bounded by the span and materialisation ceilings above
  and fails closed (never reads into or past a fabricated boundary
  further than those ceilings allow) -- but that is a bound on resource
  use and read extent, not a cryptographic proof that any given byte
  offset is the true tensor boundary.
* No model loading, no ``llama-server``/``llama.cpp`` invocation, no
  subprocess, no network.
* Every failure -- wrong magic, unsupported version, a declared length
  that exceeds the remaining file size (:attr:`GGUFMetadataStatus.TRUNCATED_OR_MALFORMED`),
  a declared count/length/nesting depth that exceeds a hard policy ceiling
  on a file that is otherwise complete (:attr:`GGUFMetadataStatus.HEADER_TOO_LARGE`),
  invalid UTF-8 in a decoded string, a missing or ambiguous
  architecture-prefixed key, an unsupported architecture, a
  non-finite/non-positive derived value, or an I/O failure on any read/seek
  after the file was opened -- is a structured, fail-closed
  :class:`GGUFArchitectureResolution`, never an exception and never a
  best-effort guess.
* Key lookup is **exact** (``f"{architecture}.block_count"`` etc.), never
  a substring/suffix match -- unlike the existing Ollama-``model_info``
  reader in :mod:`~llm_modelbench.runner` (:func:`_kv_bytes_per_token`),
  which flattens and renames keys and therefore uses a looser match. A
  direct header read has the real GGUF key names and must not blur an
  "unsupported architecture" case into a false match.

The two derivations mirrored from :func:`_kv_bytes_per_token` (kept
because they are the reason a real llama-style GGUF resolves at all, but
tightened to fail closed rather than guess):

* ``attention.head_count_kv`` absent -> falls back to
  ``attention.head_count`` (grouped-query attention collapses to
  multi-head), recorded as ``derived_kv_heads_equal_heads``.
* ``attention.key_length`` absent -> ``embedding_length // head_count``,
  recorded as ``derived_embedding_per_head`` -- but only when that
  division is exact; a non-divisible embedding means the assumption does
  not hold for this model and the resolution fails closed rather than
  rounding.
* ``attention.value_length`` absent -> equals ``key_length``, recorded as
  ``derived_value_equals_key``.

``attention.key_length`` and ``attention.value_length`` must be equal (or
made equal by the above derivation) to feed
:func:`~llm_modelbench.runtime_fit.calculate_kv_cache_bytes`, whose
``head_dimension`` input has no way to express asymmetric K/V width. An
unequal pair is refused as ``kv_asymmetric_key_value_length_unsupported``
rather than averaged, maxed, or silently narrowed to one side.

KV cache dtype is fixed at 2 bytes/scalar (f16) and ``parallel_sequences``
at 1: the managed launch command
(:func:`~llm_modelbench.llama_server_materialisation.build_llama_server_command`)
never emits ``--cache-type-k`` / ``--cache-type-v`` / ``--parallel`` /
``-np``, so llama-server's own defaults are f16 KV cache and one sequence
slot -- these constants describe what is actually launched, not an
independent guess that could disagree with it. If the managed launch ever
starts emitting those flags, this module's constants must be derived from
the same recipe value, not left as free-standing literals.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = [
    "GGUFMetadataStatus",
    "GGUFArchitectureResolution",
    "resolve_gguf_architecture",
    "KV_CACHE_DTYPE_BYTES",
    "PARALLEL_SEQUENCES",
]

#: KV cache scalar width in bytes. See module docstring: matches
#: llama-server's undeclared default (f16) because the managed launch never
#: overrides it with --cache-type-k/-v.
KV_CACHE_DTYPE_BYTES = 2

#: Concurrent sequence slots assumed for the KV estimate. See module
#: docstring: matches llama-server's undeclared default (1) because the
#: managed launch never emits --parallel/-np.
PARALLEL_SEQUENCES = 1

_MAGIC = b"GGUF"
_SUPPORTED_VERSIONS = (2, 3)
#: Hard ceiling on bytes actually *materialised* into Python objects (keys,
#: scalar values, decoded strings that are kept). A real header's kept
#: content -- key names, scalar architecture fields, a handful of short
#: strings -- is a few KiB; this generously bounds it while refusing an
#: oversized/hostile materialisation outright. Skipped array/string
#: payloads (see :data:`_MAX_SPAN_BYTES`) do not count against this.
_MAX_MATERIALISED_BYTES = 4 * 1024 * 1024
#: Hard ceiling on the total header span advanced by either a read or a
#: seek, independent of any declared length inside the file. Generously
#: covers a real file's large-but-legitimate tokenizer vocabulary arrays
#: (tens of MB) while still refusing to seek arbitrarily far into a
#: hostile declared length.
_MAX_SPAN_BYTES = 64 * 1024 * 1024
#: Hard ceiling on the number of metadata KV pairs scanned.
_MAX_METADATA_KV_COUNT = 100_000
#: Hard ceiling on any single string/array length accepted.
_MAX_ELEMENT_LEN = 1 * 1024 * 1024
#: Hard ceiling on GGUF array nesting depth (array-of-array-of-...). GGUF
#: metadata arrays are conventionally flat (array of scalars/strings); this
#: refuses a crafted deeply-nested array before Python's own recursion
#: limit could be hit. Enforced exactly: nesting to depth
#: ``_MAX_ARRAY_DEPTH`` (i.e. ``_MAX_ARRAY_DEPTH`` levels of ``_T_ARRAY``
#: wrapping a scalar/string) is accepted; one level deeper is refused.
_MAX_ARRAY_DEPTH = 8

# GGUF metadata value type tags (ggml_type / gguf value type enum).
_T_UINT8, _T_INT8 = 0, 1
_T_UINT16, _T_INT16 = 2, 3
_T_UINT32, _T_INT32 = 4, 5
_T_FLOAT32 = 6
_T_BOOL = 7
_T_STRING = 8
_T_ARRAY = 9
_T_UINT64, _T_INT64 = 10, 11
_T_FLOAT64 = 12

_SCALAR_STRUCT = {
    _T_UINT8: ("<B", 1), _T_INT8: ("<b", 1),
    _T_UINT16: ("<H", 2), _T_INT16: ("<h", 2),
    _T_UINT32: ("<I", 4), _T_INT32: ("<i", 4),
    _T_FLOAT32: ("<f", 4),
    _T_BOOL: ("<?", 1),
    _T_UINT64: ("<Q", 8), _T_INT64: ("<q", 8),
    _T_FLOAT64: ("<d", 8),
}

#: Architecture-relative keys required to derive a KV-cache estimate.
_REQUIRED_SUFFIXES = ("block_count", "attention.head_count", "embedding_length")


class GGUFMetadataStatus(str, Enum):
    """Outcome of a bounded GGUF header/metadata read."""

    RESOLVED = "resolved"
    #: The file does not start with the GGUF magic bytes.
    NOT_A_GGUF_FILE = "not_a_gguf_file"
    #: The declared GGUF version is not one this reader decodes.
    UNSUPPORTED_VERSION = "unsupported_version"
    #: The header ends, or a declared length exceeds the file's actual
    #: remaining size, before the metadata KV section could be fully read.
    #: This is real truncation/malformation, distinct from a complete file
    #: whose declared sizes merely exceed this module's own policy
    #: ceilings (see :attr:`HEADER_TOO_LARGE`).
    TRUNCATED_OR_MALFORMED = "truncated_or_malformed"
    #: A declared metadata_kv_count, string/array/element length, or array
    #: nesting depth exceeds this module's hard safety ceiling, on a file
    #: that is otherwise complete (not truncated). Distinguishes "this
    #: parser refuses to do that much work" from actual EOF/truncation.
    HEADER_TOO_LARGE = "header_too_large"
    #: The file could not be opened, or a read/seek failed at the
    #: filesystem/OS level after opening (e.g. an I/O error mid-header).
    UNREADABLE = "unreadable"
    #: No ``general.architecture`` string key was present.
    ARCHITECTURE_UNKNOWN = "architecture_unknown"
    #: One or more architecture-prefixed keys required for a KV estimate
    #: (block_count, attention.head_count, embedding_length) are absent.
    REQUIRED_METADATA_MISSING = "required_metadata_missing"
    #: A required/derived value is not a positive integer.
    INVALID_METADATA = "invalid_metadata"
    #: key_length and value_length are both present but unequal --
    #: unrepresentable by calculate_kv_cache_bytes's single head_dimension.
    KV_ASYMMETRIC_KEY_VALUE_LENGTH_UNSUPPORTED = "kv_asymmetric_key_value_length_unsupported"


@dataclass(frozen=True)
class GGUFArchitectureResolution:
    """Structured, fail-closed result of :func:`resolve_gguf_architecture`.

    ``architecture`` -- present only when ``ok`` -- is exactly the mapping
    :func:`~llm_modelbench.runtime_fit.calculate_kv_cache_bytes` expects:
    ``layer_count`` / ``kv_head_count`` / ``head_dimension`` /
    ``kv_dtype_bytes`` / ``parallel_sequences``.
    """

    status: GGUFMetadataStatus
    detail: str
    architecture_name: Optional[str] = None
    model_max_context: Optional[int] = None
    architecture: Optional[Dict[str, int]] = None
    #: Which fields were read verbatim vs. derived, for evidence. Always
    #: present (possibly empty) regardless of outcome.
    sources: Dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.sources is None:
            object.__setattr__(self, "sources", {})

    @property
    def ok(self) -> bool:
        return self.status is GGUFMetadataStatus.RESOLVED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "detail": self.detail,
            "architecture_name": self.architecture_name,
            "model_max_context": self.model_max_context,
            "architecture": dict(self.architecture) if self.architecture else None,
            "sources": dict(self.sources),
        }


class _Truncated(Exception):
    """Internal: any bounded read/seek ran past EOF, or a declared length
    exceeds the file's actual remaining size."""


class _HeaderTooLarge(Exception):
    """Internal: a declared count/length/nesting depth exceeded a hard
    policy ceiling on an otherwise-complete file."""


class _Unreadable(Exception):
    """Internal: a read/seek on the open file handle raised OSError."""


class _Reader:
    """A bounded forward-only cursor over an already-opened file handle.

    Tracks two separate budgets: ``_materialised`` (bytes actually read into
    Python objects that are kept -- keys, scalar values, decoded strings)
    against :data:`_MAX_MATERIALISED_BYTES`, and ``_span`` (total header
    bytes advanced by either a read or a skip-seek) against
    :data:`_MAX_SPAN_BYTES`. A declared length is never trusted enough to
    allocate, read, or seek on its own -- both budgets and the file's own
    remaining size are checked before any advance.
    """

    def __init__(self, handle, file_size: int) -> None:
        self._handle = handle
        self._file_size = file_size
        self._pos = 0
        self._materialised = 0
        self._span = 0

    def _advance_span(self, n: int) -> None:
        if n < 0:
            raise _Truncated("a declared length was negative")
        if self._pos + n > self._file_size:
            raise _Truncated("declared length exceeds the file's actual size")
        if self._span + n > _MAX_SPAN_BYTES:
            raise _HeaderTooLarge("header span exceeded the safety ceiling")
        self._span += n

    def read(self, n: int) -> bytes:
        if self._materialised + n > _MAX_MATERIALISED_BYTES:
            raise _HeaderTooLarge("materialised header bytes exceeded the safety ceiling")
        self._advance_span(n)
        try:
            data = self._handle.read(n)
        except OSError as exc:
            raise _Unreadable(f"read failed: {type(exc).__name__}: {exc}") from exc
        if len(data) != n:
            raise _Truncated("file ended before the declared length was satisfied")
        self._pos += n
        self._materialised += n
        return data

    def skip(self, n: int) -> None:
        """Advance past ``n`` bytes without materialising them."""
        self._advance_span(n)
        try:
            self._handle.seek(n, 1)
        except OSError as exc:
            raise _Unreadable(f"seek failed: {type(exc).__name__}: {exc}") from exc
        self._pos += n

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def gguf_string(self) -> str:
        length = self.u64()
        if length > _MAX_ELEMENT_LEN:
            raise _HeaderTooLarge("a GGUF string length exceeded the safety ceiling")
        try:
            return self.read(length).decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise _Truncated(f"a GGUF string was not valid UTF-8: {exc}") from exc

    def skip_string(self) -> None:
        length = self.u64()
        if length > _MAX_ELEMENT_LEN:
            raise _HeaderTooLarge("a GGUF string length exceeded the safety ceiling")
        self.skip(length)

    def scalar(self, value_type: int):
        fmt = _SCALAR_STRUCT.get(value_type)
        if fmt is None:
            raise _Truncated(f"unsupported GGUF scalar type tag {value_type}")
        pattern, size = fmt
        return struct.unpack(pattern, self.read(size))[0]

    def skip_value(self, value_type: int, depth: int = 0) -> None:
        """Advance past one metadata value of ``value_type`` without
        materialising its content -- used for array/string payloads not
        needed for the KV estimate. Bounded array nesting depth guards
        against a crafted array-of-array-of-... value."""
        if depth >= _MAX_ARRAY_DEPTH:
            raise _HeaderTooLarge("GGUF array nesting exceeded the safety ceiling")
        if value_type == _T_STRING:
            self.skip_string()
            return
        if value_type == _T_ARRAY:
            elem_type = self.u32()
            count = self.u64()
            if count > _MAX_ELEMENT_LEN:
                raise _HeaderTooLarge("a GGUF array length exceeded the safety ceiling")
            fmt = _SCALAR_STRUCT.get(elem_type)
            if fmt is not None:
                _, size = fmt
                self.skip(count * size)
                return
            for _ in range(count):
                self.skip_value(elem_type, depth + 1)
            return
        self.scalar(value_type)

    def value(self, value_type: int):
        """Read one metadata value of ``value_type``, materialising scalars
        and strings (both may be looked up by key) but skipping array
        payloads unmaterialised -- arrays are never consumed for the KV
        estimate, so their elements are only skipped over, not decoded."""
        if value_type == _T_STRING:
            return self.gguf_string()
        if value_type == _T_ARRAY:
            self.skip_value(value_type)
            return None
        return self.scalar(value_type)


def _lookup_int(metadata: Dict[str, Any], key: str) -> Optional[int]:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        as_int = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return as_int if as_int > 0 else None


def _read_all_metadata(reader: _Reader, metadata_kv_count: int) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for _ in range(metadata_kv_count):
        key = reader.gguf_string()
        value_type = reader.u32()
        metadata[key] = reader.value(value_type)
    return metadata


def resolve_gguf_architecture(path: str) -> GGUFArchitectureResolution:
    """Bounded, read-only extraction of the architecture inputs
    :func:`~llm_modelbench.runtime_fit.calculate_kv_cache_bytes` needs,
    from a local GGUF file's own header.

    Never raises: every failure mode is a structured
    :class:`GGUFArchitectureResolution` with ``ok is False``. Materialises
    at most :data:`_MAX_MATERIALISED_BYTES` and advances the header span by
    at most :data:`_MAX_SPAN_BYTES`, always via bounded, size-checked reads
    or skip-seeks -- the tensor data section (which follows the metadata KV
    section) is never reached.
    """
    file_path = Path(path)
    try:
        file_size = file_path.stat().st_size
        handle = file_path.open("rb")
    except OSError as exc:
        return GGUFArchitectureResolution(
            status=GGUFMetadataStatus.UNREADABLE,
            detail=f"could not open {path!r} to read GGUF metadata: {type(exc).__name__}: {exc}",
        )
    try:
        reader = _Reader(handle, file_size)
        try:
            magic = reader.read(4)
            if magic != _MAGIC:
                return GGUFArchitectureResolution(
                    status=GGUFMetadataStatus.NOT_A_GGUF_FILE,
                    detail=f"{path!r} does not start with the GGUF magic bytes",
                )
            version = reader.u32()
            if version not in _SUPPORTED_VERSIONS:
                return GGUFArchitectureResolution(
                    status=GGUFMetadataStatus.UNSUPPORTED_VERSION,
                    detail=f"GGUF version {version} is not one this reader decodes (supported: {_SUPPORTED_VERSIONS})",
                )
            _tensor_count = reader.u64()  # noqa: F841 -- read to advance the cursor, never consumed
            metadata_kv_count = reader.u64()
            if metadata_kv_count > _MAX_METADATA_KV_COUNT:
                return GGUFArchitectureResolution(
                    status=GGUFMetadataStatus.HEADER_TOO_LARGE,
                    detail=(
                        f"declared metadata_kv_count {metadata_kv_count} exceeds the "
                        f"safety ceiling of {_MAX_METADATA_KV_COUNT}"
                    ),
                )
            metadata = _read_all_metadata(reader, metadata_kv_count)
        except _Truncated as exc:
            return GGUFArchitectureResolution(
                status=GGUFMetadataStatus.TRUNCATED_OR_MALFORMED,
                detail=f"{path!r} is truncated or malformed: {exc}",
            )
        except (_HeaderTooLarge, RecursionError) as exc:
            return GGUFArchitectureResolution(
                status=GGUFMetadataStatus.HEADER_TOO_LARGE,
                detail=f"{path!r} exceeded a GGUF header safety ceiling: {exc}",
            )
        except _Unreadable as exc:
            return GGUFArchitectureResolution(
                status=GGUFMetadataStatus.UNREADABLE,
                detail=f"{path!r} became unreadable while parsing GGUF metadata: {exc}",
            )
    finally:
        try:
            handle.close()
        except OSError:
            pass

    arch_name = metadata.get("general.architecture")
    if not isinstance(arch_name, str) or not arch_name.strip():
        return GGUFArchitectureResolution(
            status=GGUFMetadataStatus.ARCHITECTURE_UNKNOWN,
            detail=f"{path!r} has no 'general.architecture' metadata key",
        )
    arch_name = arch_name.strip()
    prefix = f"{arch_name}."

    layers = _lookup_int(metadata, prefix + "block_count")
    head_count = _lookup_int(metadata, prefix + "attention.head_count")
    embedding_length = _lookup_int(metadata, prefix + "embedding_length")
    if layers is None or head_count is None or embedding_length is None:
        missing = [
            suffix for suffix, value in zip(_REQUIRED_SUFFIXES, (layers, head_count, embedding_length))
            if value is None
        ]
        return GGUFArchitectureResolution(
            status=GGUFMetadataStatus.REQUIRED_METADATA_MISSING,
            detail=(
                f"{arch_name!r} GGUF metadata is missing required key(s) for a KV "
                f"estimate: {', '.join(prefix + m for m in missing)}"
            ),
            architecture_name=arch_name,
        )

    sources: Dict[str, str] = {"layer_count": "metadata", "head_count": "metadata"}

    kv_heads = _lookup_int(metadata, prefix + "attention.head_count_kv")
    if kv_heads is None:
        kv_heads = head_count
        sources["kv_head_count"] = "derived_kv_heads_equal_heads"
    else:
        sources["kv_head_count"] = "metadata"

    key_len = _lookup_int(metadata, prefix + "attention.key_length")
    value_len = _lookup_int(metadata, prefix + "attention.value_length")
    if key_len is not None:
        sources["key_length"] = "metadata"
    if value_len is not None:
        sources["value_length"] = "metadata"

    if key_len is None:
        if embedding_length % head_count != 0:
            return GGUFArchitectureResolution(
                status=GGUFMetadataStatus.INVALID_METADATA,
                detail=(
                    f"{arch_name!r} has no {prefix}attention.key_length and "
                    f"embedding_length ({embedding_length}) is not evenly divisible "
                    f"by attention.head_count ({head_count}); refusing to guess a "
                    "rounded head dimension"
                ),
                architecture_name=arch_name,
            )
        key_len = embedding_length // head_count
        sources["key_length"] = "derived_embedding_per_head"

    if value_len is None:
        value_len = key_len
        sources["value_length"] = "derived_value_equals_key"

    if key_len != value_len:
        return GGUFArchitectureResolution(
            status=GGUFMetadataStatus.KV_ASYMMETRIC_KEY_VALUE_LENGTH_UNSUPPORTED,
            detail=(
                f"{arch_name!r} has asymmetric attention key_length ({key_len}) and "
                f"value_length ({value_len}); calculate_kv_cache_bytes has a single "
                "head_dimension input and cannot represent this without guessing"
            ),
            architecture_name=arch_name,
        )

    if min(layers, kv_heads, key_len) <= 0:
        return GGUFArchitectureResolution(
            status=GGUFMetadataStatus.INVALID_METADATA,
            detail=f"{arch_name!r} resolved a non-positive architecture dimension",
            architecture_name=arch_name,
        )

    model_max_context = _lookup_int(metadata, prefix + "context_length")

    return GGUFArchitectureResolution(
        status=GGUFMetadataStatus.RESOLVED,
        detail=f"resolved KV-estimate architecture inputs for {arch_name!r} from GGUF metadata",
        architecture_name=arch_name,
        model_max_context=model_max_context,
        architecture={
            "layer_count": layers,
            "kv_head_count": kv_heads,
            "head_dimension": key_len,
            "kv_dtype_bytes": KV_CACHE_DTYPE_BYTES,
            "parallel_sequences": PARALLEL_SEQUENCES,
        },
        sources=sources,
    )
