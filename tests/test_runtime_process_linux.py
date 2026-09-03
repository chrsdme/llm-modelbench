"""Anvil Stage 3B.3C -- Linux process-identity + teardown adapter (unit)."""
from __future__ import annotations

import pytest

from llm_modelbench import runtime_process_linux as rpl
from llm_modelbench.runtime_process_linux import (
    TerminateOutcome,
    observe_process_identity,
    terminate_process,
)


# ---------------------------------------------------------------------------
# observe_process_identity -- fixture procfs, fail-closed
# ---------------------------------------------------------------------------
def _make_proc(tmp_path, pid, *, comm="llama-server", start=4242, ppid=1,
               exe="/opt/llama.cpp/llama-server",
               cmdline=b"/opt/llama.cpp/llama-server\0--model\0/m.gguf\0"):
    entry = tmp_path / str(pid)
    entry.mkdir()
    # field 22 (start ticks) is item 18 after the state field
    (entry / "stat").write_text(f"{pid} ({comm}) S {ppid} " + "0 " * 17 + f"{start} 0\n")
    (entry / "cmdline").write_bytes(cmdline)
    if exe is not None:
        (entry / "exe_target").write_text(exe)
    return entry


def test_reads_identity_from_fixture_procfs(tmp_path, monkeypatch):
    _make_proc(tmp_path, 100)
    monkeypatch.setattr(rpl.os, "readlink", lambda p: "/opt/llama.cpp/llama-server")
    proof = observe_process_identity(100, proc_root=tmp_path)
    assert proof is not None
    assert proof.pid == 100
    assert proof.process_start_time_ticks == 4242
    assert proof.executable_path == "/opt/llama.cpp/llama-server"
    assert proof.command_argv == ("/opt/llama.cpp/llama-server", "--model", "/m.gguf")
    assert proof.parent_pid == 1


def test_comm_with_spaces_and_parentheses_is_parsed_correctly(tmp_path, monkeypatch):
    _make_proc(tmp_path, 101, comm="llama server (v2) worker")
    monkeypatch.setattr(rpl.os, "readlink", lambda p: "/x")
    proof = observe_process_identity(101, proc_root=tmp_path)
    assert proof is not None
    assert proof.process_start_time_ticks == 4242  # read after the closing ')'


def test_missing_stat_fails_closed(tmp_path):
    (tmp_path / "102").mkdir()
    assert observe_process_identity(102, proc_root=tmp_path) is None


def test_missing_exe_fails_closed(tmp_path, monkeypatch):
    _make_proc(tmp_path, 103)

    def _boom(_):
        raise OSError("permission denied")

    monkeypatch.setattr(rpl.os, "readlink", _boom)
    assert observe_process_identity(103, proc_root=tmp_path) is None


def test_empty_cmdline_fails_closed(tmp_path, monkeypatch):
    _make_proc(tmp_path, 104, cmdline=b"")
    monkeypatch.setattr(rpl.os, "readlink", lambda p: "/x")
    assert observe_process_identity(104, proc_root=tmp_path) is None


def test_malformed_stat_fails_closed(tmp_path, monkeypatch):
    entry = tmp_path / "105"
    entry.mkdir()
    (entry / "stat").write_text("garbage not a stat line")
    (entry / "cmdline").write_bytes(b"x\0")
    monkeypatch.setattr(rpl.os, "readlink", lambda p: "/x")
    assert observe_process_identity(105, proc_root=tmp_path) is None


def test_pid_mismatch_in_stat_fails_closed(tmp_path, monkeypatch):
    entry = tmp_path / "106"
    entry.mkdir()
    (entry / "stat").write_text("999 (x) S 1 " + "0 " * 17 + "10 0\n")
    (entry / "cmdline").write_bytes(b"x\0")
    monkeypatch.setattr(rpl.os, "readlink", lambda p: "/x")
    assert observe_process_identity(106, proc_root=tmp_path) is None


@pytest.mark.parametrize("bad", [0, -1, True, "5", 3.0])
def test_non_positive_pid_returns_none(bad, tmp_path):
    assert observe_process_identity(bad, proc_root=tmp_path) is None


# ---------------------------------------------------------------------------
# terminate_process -- graceful / forced / revalidation
# ---------------------------------------------------------------------------
class _Proc:
    def __init__(self, *, dies_on="terminate", start_alive=True):
        self._alive = start_alive
        self._dies_on = dies_on
        self.terminated = 0
        self.killed = 0
        self.returncode = None

    def poll(self):
        return None if self._alive else self.returncode

    def terminate(self):
        self.terminated += 1
        if self._dies_on == "terminate":
            self._alive = False
            self.returncode = -15

    def kill(self):
        self.killed += 1
        if self._dies_on in ("kill", "terminate"):
            self._alive = False
            self.returncode = -9


def _fast_clock():
    t = {"v": 0.0}

    def _now():
        t["v"] += 1.0
        return t["v"]

    return _now


def test_graceful_termination_sends_only_sigterm():
    proc = _Proc(dies_on="terminate")
    verdict = terminate_process(
        proc, graceful_timeout_s=5, forced_timeout_s=5,
        revalidate=lambda: True, monotonic=_fast_clock(), sleeper=lambda s: None,
    )
    assert verdict == TerminateOutcome.GRACEFUL
    assert proc.terminated == 1 and proc.killed == 0


def test_forced_termination_escalates_to_sigkill():
    proc = _Proc(dies_on="kill")
    verdict = terminate_process(
        proc, graceful_timeout_s=2, forced_timeout_s=5,
        revalidate=lambda: True, monotonic=_fast_clock(), sleeper=lambda s: None,
    )
    assert verdict == TerminateOutcome.FORCED
    assert proc.terminated == 1 and proc.killed == 1


def test_survives_both_signals_is_reported():
    proc = _Proc(dies_on="never")
    verdict = terminate_process(
        proc, graceful_timeout_s=2, forced_timeout_s=2,
        revalidate=lambda: True, monotonic=_fast_clock(), sleeper=lambda s: None,
    )
    assert verdict == TerminateOutcome.SURVIVED
    assert proc.terminated == 1 and proc.killed == 1


def test_already_gone_sends_no_signal():
    proc = _Proc(start_alive=False)
    proc.returncode = 0
    verdict = terminate_process(
        proc, graceful_timeout_s=2, forced_timeout_s=2,
        revalidate=lambda: pytest.fail("revalidate must not be called"),
        monotonic=_fast_clock(), sleeper=lambda s: None,
    )
    assert verdict == TerminateOutcome.ALREADY_GONE
    assert proc.terminated == 0 and proc.killed == 0


def test_failed_first_revalidation_sends_no_signal():
    proc = _Proc(dies_on="terminate")
    verdict = terminate_process(
        proc, graceful_timeout_s=2, forced_timeout_s=2,
        revalidate=lambda: False, monotonic=_fast_clock(), sleeper=lambda s: None,
    )
    assert verdict == TerminateOutcome.SURVIVED
    assert proc.terminated == 0 and proc.killed == 0


def test_second_revalidation_before_sigkill_is_enforced():
    # graceful window elapses, then the SECOND revalidation fails -> no SIGKILL
    proc = _Proc(dies_on="never")
    calls = {"n": 0}

    def _reval():
        calls["n"] += 1
        return calls["n"] == 1  # true first time (pre-SIGTERM), false pre-SIGKILL

    verdict = terminate_process(
        proc, graceful_timeout_s=2, forced_timeout_s=2,
        revalidate=_reval, monotonic=_fast_clock(), sleeper=lambda s: None,
    )
    assert calls["n"] == 2
    assert proc.terminated == 1
    assert proc.killed == 0            # SIGKILL was NOT sent
    assert verdict == TerminateOutcome.SURVIVED


def test_poll_interval_is_bounded_and_never_exceeds_window():
    assert rpl._poll_interval(0.02) <= 0.02
    assert rpl._poll_interval(100.0) == 0.05
    assert rpl._poll_interval(0.0) >= 0.001
