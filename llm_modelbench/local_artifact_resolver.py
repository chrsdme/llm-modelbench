"""Anvil Stage 3B.3E -- the smallest safe benchmark-model -> local GGUF
artifact resolver.

Stage 3B.3D left a documented blocker: the production managed ``llama-server``
spawn path has no way to turn a selected benchmark model into an authoritative
local GGUF file path plus a content-addressed SHA-256 identity, so it fails
closed at :attr:`MaterialisationStatus.RESOLVED_RECIPE_INCOMPLETE`. This module
is the narrow bridge that unblocks it.

Design (owner decisions 3B.3E-OD1 / -OD2, see
``local_only/anvil/ANVIL_PROGRESS.md``):

* Input is a single explicit model reference (the caller only reaches here
  with exactly one ``--models`` entry -- a managed ``llama-server`` serves one
  model).
* Two configuration inputs with deterministic precedence:

  1. ``gguf_artifacts`` -- an explicit ``{model_ref: absolute_path}`` mapping;
     an exact key match wins outright.
  2. ``gguf_root`` -- a single directory scanned **one level deep** (never
     recursively, never an arbitrary-tree walk). A model reference resolves
     iff **exactly one** file directly in that directory matches it.

* The resolver **never trusts a caller- or config-claimed hash**. A mapping
  value is a *path*; the returned ``verified_sha256`` is always the hash of
  the actual bytes at the resolved path (chunked, via the repository's single
  file hasher :func:`llm_modelbench.freeze._sha256`).
* Every failure -- no reference, nothing configured, a missing / unreadable /
  non-file path, more than one candidate, or an unreadable file -- is a
  structured fail-closed :class:`LocalArtifactResolution`, never an exception
  and never a silent fall-through to a broader search.

No ``subprocess``, no network, no model download / delete / conversion. The
only filesystem reads are: ``Path.exists`` / ``Path.is_file`` on an explicit
path, one non-recursive ``iterdir`` of ``gguf_root``, and a bounded chunked
read of the single resolved candidate to hash it.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from .freeze import _sha256 as _sha256_file

__all__ = [
    "LocalArtifactStatus",
    "LocalArtifactResolution",
    "resolve_local_gguf_artifact",
]

#: Filename suffixes treated as a GGUF weight artifact for the ``gguf_root``
#: match. ``.gguf`` is the only real one; the multi-part ``*-00001-of-000NN``
#: naming still ends in ``.gguf`` so a plain suffix check is sufficient. An
#: ``mmproj`` companion is deliberately *not* matched here -- 3B.3E resolves
#: the primary weight file only.
_GGUF_SUFFIXES = (".gguf",)


class LocalArtifactStatus(str, Enum):
    """Outcome of a local-artifact resolution attempt."""

    RESOLVED = "resolved"
    #: The caller had no single explicit model reference to resolve (multi
    #: model, all-installed default, or ``--select``). Not an error -- the
    #: managed spawn simply stays fail-closed.
    NO_MODEL_REF = "no_model_ref"
    #: Neither ``gguf_artifacts`` nor ``gguf_root`` is configured.
    NOT_CONFIGURED = "not_configured"
    #: A configured path (map entry or the sole root match) does not exist.
    MISSING = "missing"
    #: A configured path exists but could not be read / stat-ed.
    UNREADABLE = "unreadable"
    #: A configured path exists but is a directory (or other non-regular file).
    NOT_A_FILE = "not_a_file"
    #: ``gguf_root`` contains more than one file matching the model reference.
    AMBIGUOUS = "ambiguous"
    #: The resolved candidate file could not be hashed (read error mid-stream).
    HASH_FAILED = "hash_failed"


@dataclass(frozen=True)
class LocalArtifactResolution:
    """Structured, fail-closed result of :func:`resolve_local_gguf_artifact`.

    ``verified_sha256`` -- when present -- is the bare lowercase hex digest of
    the *actual bytes* at :attr:`resolved_path`, never a value copied from
    configuration or a caller.
    """

    status: LocalArtifactStatus
    model_ref: Optional[str]
    detail: str
    resolved_path: Optional[str] = None
    verified_sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    #: Every candidate considered for an ``AMBIGUOUS`` outcome (sorted), so the
    #: operator can see exactly what collided. Empty otherwise.
    candidate_paths: Tuple[str, ...] = ()
    #: Which configuration input produced the resolution ("map" / "root"),
    #: recorded for evidence. ``None`` when nothing was resolved.
    source: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status is LocalArtifactStatus.RESOLVED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "model_ref": self.model_ref,
            "detail": self.detail,
            "resolved_path": self.resolved_path,
            "verified_sha256": self.verified_sha256,
            "size_bytes": self.size_bytes,
            "candidate_paths": list(self.candidate_paths),
            "source": self.source,
        }


def _hash_regular_file(
    path: Path, hasher: Callable[[Path], str]
) -> "tuple[Optional[str], Optional[int], Optional[LocalArtifactResolution]]":
    """Hash ``path`` (already known to exist). Returns
    ``(digest, size_bytes, None)`` on success or ``(None, None, failure)``.
    """
    try:
        if not path.is_file():
            return None, None, None  # signalled by caller as NOT_A_FILE
        size = path.stat().st_size
    except OSError as exc:
        return (
            None,
            None,
            LocalArtifactResolution(
                status=LocalArtifactStatus.UNREADABLE,
                model_ref=None,
                detail=f"could not stat {str(path)!r}: {type(exc).__name__}: {exc}",
                resolved_path=str(path),
            ),
        )
    try:
        digest = hasher(path)
    except OSError as exc:
        return (
            None,
            None,
            LocalArtifactResolution(
                status=LocalArtifactStatus.HASH_FAILED,
                model_ref=None,
                detail=(
                    f"could not hash the bytes at {str(path)!r} to prove "
                    f"artifact identity: {type(exc).__name__}: {exc}"
                ),
                resolved_path=str(path),
            ),
        )
    return _normalise_hex(digest), int(size), None


def _normalise_hex(value: str) -> str:
    v = str(value).strip().lower()
    if v.startswith("sha256:"):
        v = v[len("sha256:") :]
    return v


def _matches_model_ref(filename: str, model_ref: str) -> bool:
    """A ``gguf_root`` file matches the model reference when its name (with or
    without the ``.gguf`` suffix) equals the reference, or the reference is a
    substring of the stem. Deliberately simple and case-sensitive -- the
    operator controls both the directory contents and the ``--models`` value,
    and any collision is reported as ``AMBIGUOUS`` rather than silently
    picked."""
    lower = filename.lower()
    if not lower.endswith(_GGUF_SUFFIXES):
        return False
    stem = filename
    for suffix in _GGUF_SUFFIXES:
        if lower.endswith(suffix):
            stem = filename[: -len(suffix)]
            break
    return model_ref == filename or model_ref == stem or model_ref in stem


def resolve_local_gguf_artifact(
    model_ref: Optional[str],
    *,
    artifacts_map: Optional[Mapping[str, str]] = None,
    root_dir: Optional[str] = None,
    hasher: Callable[[Path], str] = _sha256_file,
) -> LocalArtifactResolution:
    """Resolve ``model_ref`` to a local GGUF path + verified SHA-256.

    See the module docstring for the precedence and fail-closed contract.
    """
    artifacts_map = dict(artifacts_map or {})
    ref = model_ref.strip() if isinstance(model_ref, str) else ""
    if not ref:
        return LocalArtifactResolution(
            status=LocalArtifactStatus.NO_MODEL_REF,
            model_ref=None,
            detail=(
                "no single explicit benchmark model reference was available to "
                "resolve a local artifact for (a managed llama-server serves "
                "one model; multi-model, all-installed, and --select runs do "
                "not reach the managed path)"
            ),
        )

    have_map = bool(artifacts_map)
    have_root = isinstance(root_dir, str) and root_dir.strip() != ""
    if not have_map and not have_root:
        return LocalArtifactResolution(
            status=LocalArtifactStatus.NOT_CONFIGURED,
            model_ref=ref,
            detail=(
                "no local GGUF artifact source is configured; set "
                "'gguf_artifacts' (an explicit {model: path} map) or "
                "'gguf_root' (a directory) to enable the managed llama-server "
                "path"
            ),
        )

    # --- precedence 1: explicit per-model mapping -----------------------
    if ref in artifacts_map:
        raw = artifacts_map[ref]
        path = Path(str(raw))
        if not path.exists():
            return LocalArtifactResolution(
                status=LocalArtifactStatus.MISSING,
                model_ref=ref,
                detail=(
                    f"gguf_artifacts[{ref!r}] points at {str(path)!r}, which "
                    f"does not exist"
                ),
                resolved_path=str(path),
                source="map",
            )
        digest, size, failure = _hash_regular_file(path, hasher)
        if failure is not None:
            return LocalArtifactResolution(
                status=failure.status,
                model_ref=ref,
                detail=failure.detail,
                resolved_path=str(path),
                source="map",
            )
        if digest is None:
            return LocalArtifactResolution(
                status=LocalArtifactStatus.NOT_A_FILE,
                model_ref=ref,
                detail=(
                    f"gguf_artifacts[{ref!r}] points at {str(path)!r}, which is "
                    f"not a regular file"
                ),
                resolved_path=str(path),
                source="map",
            )
        return LocalArtifactResolution(
            status=LocalArtifactStatus.RESOLVED,
            model_ref=ref,
            detail=f"resolved {ref!r} to {str(path)!r} via gguf_artifacts",
            resolved_path=str(path),
            verified_sha256=digest,
            size_bytes=size,
            source="map",
        )

    # --- precedence 2: single non-recursive directory scan -------------
    if not have_root:
        return LocalArtifactResolution(
            status=LocalArtifactStatus.MISSING,
            model_ref=ref,
            detail=(
                f"{ref!r} is not in gguf_artifacts and no gguf_root is "
                f"configured to search"
            ),
            source="map",
        )

    root = Path(str(root_dir))
    try:
        if not root.is_dir():
            return LocalArtifactResolution(
                status=LocalArtifactStatus.MISSING,
                model_ref=ref,
                detail=f"gguf_root {str(root)!r} is not a directory",
                resolved_path=str(root),
                source="root",
            )
        entries = sorted(
            entry
            for entry in root.iterdir()
            if entry.is_file() and _matches_model_ref(entry.name, ref)
        )
    except OSError as exc:
        return LocalArtifactResolution(
            status=LocalArtifactStatus.UNREADABLE,
            model_ref=ref,
            detail=(
                f"could not list gguf_root {str(root)!r}: "
                f"{type(exc).__name__}: {exc}"
            ),
            resolved_path=str(root),
            source="root",
        )

    if not entries:
        return LocalArtifactResolution(
            status=LocalArtifactStatus.MISSING,
            model_ref=ref,
            detail=(
                f"no file in gguf_root {str(root)!r} matches {ref!r} "
                f"(non-recursive, {' / '.join(_GGUF_SUFFIXES)} only)"
            ),
            resolved_path=str(root),
            source="root",
        )
    if len(entries) > 1:
        return LocalArtifactResolution(
            status=LocalArtifactStatus.AMBIGUOUS,
            model_ref=ref,
            detail=(
                f"{len(entries)} files in gguf_root {str(root)!r} match {ref!r}; "
                f"add an explicit gguf_artifacts[{ref!r}] entry to disambiguate"
            ),
            resolved_path=str(root),
            candidate_paths=tuple(str(p) for p in entries),
            source="root",
        )

    path = entries[0]
    digest, size, failure = _hash_regular_file(path, hasher)
    if failure is not None:
        return LocalArtifactResolution(
            status=failure.status,
            model_ref=ref,
            detail=failure.detail,
            resolved_path=str(path),
            source="root",
        )
    # A directory entry that passed ``is_file()`` in the scan but fails the
    # re-check in ``_hash_regular_file`` (raced away) -> fail closed.
    if digest is None:
        return LocalArtifactResolution(
            status=LocalArtifactStatus.NOT_A_FILE,
            model_ref=ref,
            detail=f"{str(path)!r} is no longer a regular file",
            resolved_path=str(path),
            source="root",
        )
    return LocalArtifactResolution(
        status=LocalArtifactStatus.RESOLVED,
        model_ref=ref,
        detail=f"resolved {ref!r} to {str(path)!r} via gguf_root",
        resolved_path=str(path),
        verified_sha256=digest,
        size_bytes=size,
        source="root",
    )
