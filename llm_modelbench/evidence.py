"""Anvil Stage 0 — evidence model: typed results, an append-only ledger with
generic provenance, a versioned projection store, and the locking/atomicity
primitives both are built on.

Two storage primitives, deliberately kept distinct (see
ANVIL_MASTER_PLAN.md Stage 0.2 and the "What changed from v1" note there --
v1 conflated these into one mutable "current pointer on evidence itself"
design, which was wrong):

- :class:`EvidenceLedger` -- append-only, for facts: a primary model output,
  a recovery attempt, a judge result, a capability observation, a hardware
  snapshot, a runtime event. These never become "not current" in the sense
  of disappearing -- they are permanently true of the moment recorded.
- :class:`ProjectionStore` -- for views computed *from* the ledger: "the
  current compatible capability set," "the current model-card summary,"
  "the current canonical ranking release." Only projections get a movable
  current pointer; raw evidence never does. This is the actual mechanism
  behind full-unattended-adoption-with-reversibility: "adopt" is a pointer
  move here, not a mutation of evidence.

Nothing in this module is wired into runner.py/campaign.py/rankings.py yet
-- that integration is later-stage work (Stage 2 onward) per the plan's own
dependency order. Stage 0's job is a solid, well-tested foundation.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import socket
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Typed measurement results (Stage 0.3)
# ---------------------------------------------------------------------------


class EvalStatus(str, Enum):
    """Closes a real ambiguity: a bare ``score: 0`` is insufficient by
    itself to distinguish a measured zero from an environment, capability,
    or harness disposition. The existing pipeline already carries some of
    these distinctions ad hoc (e.g. ``kv_cache_exceeds_vram_budget`` as its
    own status string in current gap reports, confirmed by direct
    inspection during Stage 0.0) -- this makes it structural."""

    MEASURED = "measured"
    ENVIRONMENT_SKIPPED = "environment_skipped"
    CAPABILITY_UNSUPPORTED = "capability_unsupported"
    CAPABILITY_INCONCLUSIVE = "capability_inconclusive"
    HARNESS_ERROR = "harness_error"
    AWAITING_JUDGE = "awaiting_judge"
    OPERATOR_EXCLUDED = "operator_excluded"


@dataclass(frozen=True)
class EvalResult:
    status: EvalStatus
    score: Optional[float] = None
    named_metrics: Mapping[str, float] = field(default_factory=dict)
    reason: Optional[str] = None
    evidence_refs: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()
    error_kind: Optional[str] = None
    scorer_version: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status is EvalStatus.MEASURED and self.score is None:
            raise ValueError("EvalStatus.MEASURED requires a non-None score")


@dataclass(frozen=True)
class QualityMeasurement:
    """Correctness/accuracy dimension only. Never blended with performance
    or environment numbers into one mysterious "score" -- see
    ANVIL_MASTER_PLAN.md Stage 0.3."""

    score: Optional[float]
    named_metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PerformanceMeasurement:
    tokens_per_second: Optional[float] = None
    time_to_first_token_ms: Optional[float] = None
    inter_token_latency_p50_ms: Optional[float] = None
    inter_token_latency_p95_ms: Optional[float] = None
    wall_seconds: Optional[float] = None


@dataclass(frozen=True)
class EnvironmentMeasurement:
    vram_peak_mb: Optional[float] = None
    offload_fraction: Optional[float] = None
    power_mean_w: Optional[float] = None
    temp_peak_c: Optional[float] = None
    context_length_used: Optional[int] = None


class EvidenceTrustClass(str, Enum):
    """Stage 0.1's answer to "legacy evidence becomes read-only, but read-
    only is not the same as trusted-for-current-comparison." Prevents a
    legacy adapter from silently promoting old evidence (old scorers, old
    task hashes, known-broken capability routing, debug/calibration runs)
    into current model-card truth."""

    CANONICAL_COMPATIBLE = "canonical_compatible"
    HISTORICAL_VALID = "historical_valid"
    CALIBRATION_ONLY = "calibration_only"
    KNOWN_INVALID = "known_invalid"
    UNKNOWN_LEGACY = "unknown_legacy"


# ---------------------------------------------------------------------------
# Locking / atomicity primitives (Stage 0.2)
# ---------------------------------------------------------------------------


def new_event_id(record_type: str, payload: Mapping[str, Any]) -> str:
    """Content-addressed, not random: appending the same logical event twice
    (identical type + payload) is naturally idempotent -- the second append
    is a no-op (see :meth:`EvidenceLedger.append`). Callers that need two
    distinct records with identical payloads must include a distinguishing
    field in the payload themselves (e.g. an explicit attempt counter) --
    this is deliberate, not an oversight: silent accidental dedup of
    genuinely-different events would be worse than requiring an explicit
    nonce for the rare case that needs one.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256((record_type + "\0" + canonical).encode("utf-8")).hexdigest()
    return digest[:24]


def atomic_write_text(path: Path, text: str) -> None:
    """Write-to-temp-then-rename. os.replace is atomic on POSIX and Windows,
    so no reader ever observes a partially-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + f".tmp.{os.getpid()}.{time.time_ns()}")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, sort_keys=True, indent=2) + "\n")


class StaleLockError(RuntimeError):
    """Raised when a lock file exists but its owning process is
    unambiguously gone, rather than silently stealing or silently blocking
    forever. The old campaign system learned this lesson (PID/hostname/
    phase-aware stale-lock handling) -- keep it, simplify the
    implementation."""


class LockHeldError(RuntimeError):
    """Raised when a lock is held by a live process (or a process we cannot
    prove is dead) and ``blocking=False`` was requested."""


@dataclass(frozen=True)
class _LockInfo:
    pid: int
    hostname: str
    acquired_at: float
    phase: str


class FileLock:
    """A single-writer lock backed by an exclusively-created lock file
    (``O_EXCL`` -- atomic create-or-fail, no TOCTOU window). Stale locks
    (owning PID no longer running on this host) are detected and reclaimed
    rather than trusted or ignored; a lock held by a different, live host
    is never assumed stale.
    """

    def __init__(self, path: Path, *, phase: str = "default"):
        self.path = path
        self.phase = phase
        self._held = False

    def _read_owner(self) -> Optional[_LockInfo]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        try:
            return _LockInfo(
                pid=int(data["pid"]),
                hostname=str(data["hostname"]),
                acquired_at=float(data["acquired_at"]),
                phase=str(data.get("phase", "")),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _owner_is_dead(self, owner: _LockInfo) -> bool:
        if owner.hostname != socket.gethostname():
            return False  # never assume a different host's lock is stale
        try:
            os.kill(owner.pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False  # process exists, just not ours to signal
        return False

    def acquire(self, *, blocking: bool = True, timeout: float = 30.0, poll_interval: float = 0.1) -> None:
        deadline = time.monotonic() + timeout
        while True:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(
                {
                    "pid": os.getpid(),
                    "hostname": socket.gethostname(),
                    "acquired_at": time.time(),
                    "phase": self.phase,
                }
            )
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(fd, payload.encode("utf-8"))
                finally:
                    os.close(fd)
                self._held = True
                return
            except FileExistsError:
                owner = self._read_owner()
                if owner is not None and self._owner_is_dead(owner):
                    try:
                        self.path.unlink()
                    except FileNotFoundError:
                        pass  # another racer already reclaimed it -- fine, retry
                    except OSError as exc:
                        # Owner is unambiguously dead but we still couldn't
                        # clear the lock file (e.g. a permissions problem) --
                        # surface this distinctly rather than looping forever
                        # or silently falling through to LockHeldError, which
                        # would misleadingly imply a live owner.
                        raise StaleLockError(f"{self.path}: {exc}") from exc
                    continue  # retry immediately, we just cleared a stale lock
                if not blocking:
                    raise LockHeldError(str(self.path))
                if time.monotonic() >= deadline:
                    raise LockHeldError(f"timed out waiting for {self.path}")
                time.sleep(poll_interval)

    def release(self) -> None:
        if self._held:
            with contextlib.suppress(FileNotFoundError):
                self.path.unlink()
            self._held = False

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


# ---------------------------------------------------------------------------
# Immutable evidence ledger + generic provenance (Stage 0.2)
# ---------------------------------------------------------------------------


class ProvenanceRelation(str, Enum):
    DERIVED_FROM = "derived_from"
    SUPERSEDES = "supersedes"
    JUDGES = "judges"
    MEASURES = "measures"
    RETRIES = "retries"


@dataclass(frozen=True)
class ProvenanceLink:
    relation: ProvenanceRelation
    target_record_id: str


@dataclass(frozen=True)
class LedgerRecord:
    record_id: str
    record_type: str
    payload: Mapping[str, Any]
    provenance: Tuple[ProvenanceLink, ...] = ()
    trust_class: EvidenceTrustClass = EvidenceTrustClass.CANONICAL_COMPATIBLE

    def to_json_line(self) -> str:
        return json.dumps(
            {
                "record_id": self.record_id,
                "record_type": self.record_type,
                "payload": self.payload,
                "provenance": [
                    {"relation": link.relation.value, "target_record_id": link.target_record_id}
                    for link in self.provenance
                ],
                "trust_class": self.trust_class.value,
            },
            sort_keys=True,
        )

    @classmethod
    def from_json_line(cls, line: str) -> "LedgerRecord":
        data = json.loads(line)
        return cls(
            record_id=data["record_id"],
            record_type=data["record_type"],
            payload=data["payload"],
            provenance=tuple(
                ProvenanceLink(ProvenanceRelation(p["relation"]), p["target_record_id"])
                for p in data.get("provenance", [])
            ),
            trust_class=EvidenceTrustClass(data.get("trust_class", EvidenceTrustClass.CANONICAL_COMPATIBLE.value)),
        )


class EvidenceLedgerError(RuntimeError):
    pass


class EvidenceLedger:
    """Append-only, JSONL-backed. Every append is atomic (write-temp,
    ``os.replace``) and idempotent by content-addressed ``record_id``
    (see :func:`new_event_id`). No update or delete method exists by
    design -- that is the immutability guarantee, not an omission."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = FileLock(path.with_suffix(path.suffix + ".lock"), phase="append")
        self._index: Optional[Dict[str, LedgerRecord]] = None
        self._malformed_lines: List[Tuple[int, str]] = []

    def _load_index(self) -> Dict[str, LedgerRecord]:
        if self._index is not None:
            return self._index
        index: Dict[str, LedgerRecord] = {}
        malformed: List[Tuple[int, str]] = []
        if self.path.exists():
            for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    record = LedgerRecord.from_json_line(line)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    # A malformed line is skipped -- never treated as a valid
                    # record -- rather than failing the whole ledger read.
                    # Every consumer of get/all/find inherits this: fail-closed
                    # means an unusable line drops out of consideration, not
                    # that the entire ledger becomes unreadable.
                    malformed.append((line_number, f"{type(exc).__name__}: {exc}"))
                    continue
                index[record.record_id] = record
        self._index = index
        self._malformed_lines = malformed
        return index

    def malformed_lines(self) -> Tuple[Tuple[int, str], ...]:
        """1-based line numbers and reasons for lines skipped during the
        most recent index load because they failed to parse as a
        ``LedgerRecord``. Empty when the ledger is clean. Call after any
        read (``get``/``all``/``find``) to check for corrupt entries."""
        self._load_index()
        return tuple(self._malformed_lines)

    def append(
        self,
        record_type: str,
        payload: Mapping[str, Any],
        *,
        provenance: Sequence[ProvenanceLink] = (),
        trust_class: EvidenceTrustClass = EvidenceTrustClass.CANONICAL_COMPATIBLE,
        record_id: Optional[str] = None,
    ) -> LedgerRecord:
        record_id = record_id or new_event_id(record_type, payload)
        with self._lock:
            index = self._load_index()
            existing = index.get(record_id)
            if existing is not None:
                if existing.record_type != record_type or dict(existing.payload) != dict(payload):
                    raise EvidenceLedgerError(
                        f"record_id {record_id!r} already exists with different content "
                        "(content-addressed id collision on genuinely different events "
                        "should not happen -- if intentional, pass an explicit distinct "
                        "record_id)"
                    )
                return existing  # idempotent: identical event already recorded
            record = LedgerRecord(
                record_id=record_id,
                record_type=record_type,
                payload=dict(payload),
                provenance=tuple(provenance),
                trust_class=trust_class,
            )
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(record.to_json_line() + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            index[record_id] = record
            return record

    def get(self, record_id: str) -> Optional[LedgerRecord]:
        return self._load_index().get(record_id)

    def all(self) -> Iterator[LedgerRecord]:
        return iter(self._load_index().values())

    def find(self, *, record_type: Optional[str] = None) -> List[LedgerRecord]:
        return [r for r in self.all() if record_type is None or r.record_type == record_type]


class EffectiveEvidenceResolutionError(RuntimeError):
    pass


class EffectiveEvidenceResolver:
    """Walks a ledger's ``supersedes`` links to find the currently-
    authoritative terminal record for a lineage, without deleting any
    history. Replaces the old campaign system's special-purpose
    supersession-graph machinery *in principle* -- kept generic here
    (arbitrary relation types, not just supersession) rather than
    reinventing a dedicated subsystem.

    Design checkpoint (recorded, not just implemented): this needs to earn
    its "simpler than what it replaced" claim once Stage 4 actually uses it
    for real campaign data, not just assert it because the class is short.
    """

    def __init__(self, ledger: EvidenceLedger):
        self.ledger = ledger

    def resolve(self, record_id: str) -> LedgerRecord:
        """Follow SUPERSEDES edges starting at record_id to the terminal
        (non-superseded) record. Raises on a cycle, a missing source, or an
        ambiguous terminal (more than one record claims to supersede the
        same predecessor -- a fork)."""
        successors_by_target: Dict[str, List[str]] = {}
        for record in self.ledger.all():
            for link in record.provenance:
                if link.relation is ProvenanceRelation.SUPERSEDES:
                    successors_by_target.setdefault(link.target_record_id, []).append(record.record_id)

        current = self.ledger.get(record_id)
        if current is None:
            if record_id in successors_by_target:
                # Some existing record claims to supersede record_id, but
                # record_id itself was never actually appended -- a broken
                # reference, distinct from "you asked about an id nobody
                # has ever heard of."
                raise EffectiveEvidenceResolutionError(
                    f"missing source: {successors_by_target[record_id]!r} claims to "
                    f"supersede {record_id!r}, which is not present in the ledger"
                )
            raise EffectiveEvidenceResolutionError(f"unknown record_id: {record_id!r}")

        seen: List[str] = []
        while True:
            if current.record_id in seen:
                raise EffectiveEvidenceResolutionError(
                    f"cycle detected in supersession chain: {seen + [current.record_id]}"
                )
            seen.append(current.record_id)
            successors = successors_by_target.get(current.record_id, [])
            if not successors:
                return current
            if len(successors) > 1:
                raise EffectiveEvidenceResolutionError(
                    f"ambiguous terminal: {current.record_id!r} is superseded by "
                    f"more than one record ({successors}) -- forked replacement chain"
                )
            # successors[0] was discovered by iterating real ledger records
            # above, so it is always fetchable here -- no further existence
            # check needed (a broken/dangling SUPERSEDES target is caught
            # by the record_id-not-in-ledger branch above instead).
            current = self.ledger.get(successors[0])


# ---------------------------------------------------------------------------
# Versioned projection + movable current pointer (Stage 0.2)
# ---------------------------------------------------------------------------


class ProjectionStore:
    """Views computed *from* the ledger, never raw evidence. Every write is
    a new immutable version; only the *pointer* to "current" moves, and
    moving it is a single atomic file replace. This is the actual mechanism
    behind reversible unattended adoption: a bad adoption is
    ``set_current`` back to the previous version_id, not a rebuild.
    """

    def __init__(self, root: Path):
        self.root = root

    def _versions_dir(self, projection_id: str) -> Path:
        return self.root / projection_id / "versions"

    def _pointer_path(self, projection_id: str) -> Path:
        return self.root / projection_id / "current.json"

    def write_version(self, projection_id: str, payload: Any) -> str:
        """Writes a new immutable version; does not move the current
        pointer. Returns the version_id."""
        versions_dir = self._versions_dir(projection_id)
        versions_dir.mkdir(parents=True, exist_ok=True)
        version_id = new_event_id(f"projection:{projection_id}", {"payload": payload, "n": time.time_ns()})
        atomic_write_json(versions_dir / f"{version_id}.json", payload)
        return version_id

    def set_current(self, projection_id: str, version_id: str) -> None:
        if not (self._versions_dir(projection_id) / f"{version_id}.json").exists():
            raise FileNotFoundError(f"no such version {version_id!r} for projection {projection_id!r}")
        atomic_write_json(self._pointer_path(projection_id), {"version_id": version_id, "set_at": time.time()})

    def get_current_version_id(self, projection_id: str) -> Optional[str]:
        pointer_path = self._pointer_path(projection_id)
        if not pointer_path.exists():
            return None
        return json.loads(pointer_path.read_text(encoding="utf-8"))["version_id"]

    def get_current(self, projection_id: str) -> Optional[Any]:
        version_id = self.get_current_version_id(projection_id)
        if version_id is None:
            return None
        return self.get_version(projection_id, version_id)

    def get_version(self, projection_id: str, version_id: str) -> Any:
        path = self._versions_dir(projection_id) / f"{version_id}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def rollback(self, projection_id: str, to_version_id: str) -> None:
        """Alias for set_current, named for the reversible-adoption use
        case: moving the pointer backward is exactly as cheap and exactly
        the same operation as moving it forward."""
        self.set_current(projection_id, to_version_id)
