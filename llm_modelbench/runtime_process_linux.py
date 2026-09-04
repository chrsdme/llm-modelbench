"""Anvil Stage 3B.3C -- Linux process-identity + teardown adapter.

The pure lifecycle domain (:mod:`~llm_modelbench.runtime_lifecycle`) defines
*what* must be proven before a destructive action; this module is the
platform adapter that actually reads ``/proc`` and sends signals. It is
deliberately separate so the domain stays free of ``subprocess`` / ``signal``
/ procfs mechanics.

Design points:

* **No new proc-stat parser.** ``telemetry.parse_proc_stat`` already parses a
  Linux ``/proc/<pid>/stat`` record correctly -- the process name in
  ``comm`` may contain spaces or parentheses, so field 22 (start ticks) is
  read *after* the closing ``)`` of ``comm``, never by naive whitespace
  splitting. This module reuses it.
* **Fail closed.** A missing/again-unreadable procfs field yields ``None``
  from :func:`observe_process_identity`, which the lifecycle controller
  treats as "identity could not be revalidated" -> cleanup refused.
* **Scoped teardown only.** :func:`terminate_process` signals exactly one PID
  (the direct child ModelBench launched) after the caller has revalidated
  identity. No ``pkill`` / ``killall`` / name-based kill / process-group
  kill / port-based kill. The launched ``llama-server`` is a direct child
  (no ``start_new_session``), so ``proc.terminate()`` / ``proc.kill()`` on
  the retained :class:`subprocess.Popen` object is sufficient and is not
  vulnerable to PID reuse (the OS keeps the child handle until reaped).
"""
from __future__ import annotations

import os
import time
from enum import Enum
from pathlib import Path
from typing import Optional

from .runtime_lifecycle import LaunchProcessProof
from .telemetry import parse_proc_stat

__all__ = [
    "PROC_ROOT",
    "observe_process_identity",
    "read_process_rss_bytes",
    "terminate_process",
    "TerminateOutcome",
]

PROC_ROOT = Path("/proc")

#: Bound on a single ``/proc`` file read (matches ``telemetry`` conventions).
_MAX_PROC_FILE_BYTES = 64 * 1024
_MAX_CMDLINE_BYTES = 8 * 1024
_MAX_CMDLINE_ARGS = 64


def _read_bounded(path: Path, limit: int) -> Optional[bytes]:
    try:
        with path.open("rb") as handle:
            data = handle.read(limit + 1)
    except OSError:
        return None
    if len(data) > limit:
        return None
    return data


def observe_process_identity(
    pid: int,
    *,
    proc_root: Path = PROC_ROOT,
) -> Optional[LaunchProcessProof]:
    """Re-observe the live process identity for ``pid`` from procfs.

    Returns a :class:`LaunchProcessProof` on success, or ``None`` if any
    required field (``stat`` start ticks, ``exe`` target, ``cmdline``) cannot
    be read or parsed -- the lifecycle controller treats ``None`` as
    "identity could not be revalidated" and refuses destructive cleanup.

    This never sends a signal and never mutates anything.
    """
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return None
    entry = proc_root / str(pid)

    raw_stat = _read_bounded(entry / "stat", _MAX_PROC_FILE_BYTES)
    if raw_stat is None:
        return None
    try:
        stat_pid, ppid, start_ticks = parse_proc_stat(raw_stat.decode("utf-8", "replace"))
    except ValueError:
        return None
    if stat_pid != pid:
        # PID changed under us mid-read -> cannot prove identity.
        return None

    try:
        executable_path = os.readlink(entry / "exe")
    except OSError:
        # A live process we own always has a readable /proc/<pid>/exe. Its
        # absence (permission, or the process is gone) means we cannot prove
        # identity -> fail closed.
        return None

    raw_cmdline = _read_bounded(entry / "cmdline", _MAX_CMDLINE_BYTES)
    if raw_cmdline is None:
        return None
    if raw_cmdline == b"":
        # A kernel thread or a process that scrubbed its cmdline -- not our
        # llama-server; cannot match its recorded argv.
        return None
    if not raw_cmdline.endswith(b"\0"):
        return None
    parts = raw_cmdline.split(b"\0")[:-1]
    if len(parts) > _MAX_CMDLINE_ARGS:
        return None
    try:
        command_argv = tuple(part.decode("utf-8", "strict") for part in parts)
    except UnicodeDecodeError:
        return None

    try:
        return LaunchProcessProof(
            pid=pid,
            process_start_time_ticks=start_ticks,
            executable_path=executable_path,
            command_argv=command_argv,
            parent_pid=ppid,
        )
    except ValueError:
        return None


def read_process_rss_bytes(pid: int, *, proc_root: Path = PROC_ROOT) -> Optional[int]:
    """Anvil Stage 3B.5 -- a bounded, one-shot read of ``VmRSS`` from
    ``/proc/<pid>/status`` for evidence (never a scoring input).

    Same discipline as :func:`observe_process_identity`: bounded read, no
    signal sent, no mutation, ``None`` on anything unreadable/unparsable
    (process gone, permission denied, field absent, non-numeric) -- a
    resident-set read failing is evidence of "not observed", never a reason
    to fail the caller.
    """
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return None
    raw = _read_bounded(proc_root / str(pid) / "status", _MAX_PROC_FILE_BYTES)
    if raw is None:
        return None
    try:
        text = raw.decode("utf-8", "replace")
    except UnicodeDecodeError:  # pragma: no cover -- decode("replace") never raises
        return None
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) != 3 or parts[2] != "kB":
                return None
            try:
                kib = int(parts[1])
            except ValueError:
                return None
            if kib < 0:
                return None
            return kib * 1024
    return None


class TerminateOutcome(str, Enum):
    """Result of a single scoped termination attempt."""

    #: The process exited within the graceful (SIGTERM) window.
    GRACEFUL = "graceful"
    #: The process required the forced (SIGKILL) signal and then exited.
    FORCED = "forced"
    #: The process survived both signals within their bounded windows.
    SURVIVED = "survived"
    #: The process was already gone before any signal was sent.
    ALREADY_GONE = "already_gone"


def _still_running(proc) -> bool:
    return proc.poll() is None


def terminate_process(
    proc,
    *,
    graceful_timeout_s: float,
    forced_timeout_s: float,
    revalidate,
    monotonic=time.monotonic,
    sleeper=time.sleep,
) -> str:
    """Terminate exactly the direct child ``proc`` (a :class:`subprocess.Popen`).

    ``revalidate()`` is called with no arguments and must return ``True`` only
    if the caller can *still* prove ``proc`` is the process ModelBench
    launched (anti-PID-reuse). It is consulted **twice** -- once before
    SIGTERM and, if the graceful window elapses, **again** before SIGKILL
    (the process occupying that PID after SIGTERM is not assumed to still be
    ours merely because it was proven before). A ``False`` at either point
    aborts without sending that signal.

    Returns a :class:`TerminateOutcome` value. Raises nothing for the normal
    outcomes; an ``OSError`` from an unexpected signal failure propagates.
    """
    if not _still_running(proc):
        return TerminateOutcome.ALREADY_GONE

    if not revalidate():
        # Cannot prove ownership -> send nothing. Fail closed.
        return TerminateOutcome.SURVIVED

    try:
        proc.terminate()  # SIGTERM
    except ProcessLookupError:
        return TerminateOutcome.ALREADY_GONE

    deadline = monotonic() + max(0.0, float(graceful_timeout_s))
    while monotonic() < deadline:
        if not _still_running(proc):
            return TerminateOutcome.GRACEFUL
        sleeper(_poll_interval(graceful_timeout_s))
    if not _still_running(proc):
        return TerminateOutcome.GRACEFUL

    # Graceful window elapsed. Re-prove ownership before the forced signal.
    if not revalidate():
        return TerminateOutcome.SURVIVED

    try:
        proc.kill()  # SIGKILL
    except ProcessLookupError:
        return TerminateOutcome.FORCED

    deadline = monotonic() + max(0.0, float(forced_timeout_s))
    while monotonic() < deadline:
        if not _still_running(proc):
            return TerminateOutcome.FORCED
        sleeper(_poll_interval(forced_timeout_s))
    if not _still_running(proc):
        return TerminateOutcome.FORCED

    return TerminateOutcome.SURVIVED


def _poll_interval(timeout_s: float) -> float:
    """A small bounded poll interval, never larger than the window itself."""
    return min(0.05, max(0.001, float(timeout_s) / 20.0))
