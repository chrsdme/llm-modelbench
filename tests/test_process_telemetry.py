import os

import pytest

from llm_modelbench import process_telemetry as pt
from llm_modelbench.telemetry import CommandResult, TelemetryFieldState


STAMP = "2026-08-01T12:00:00Z"


def _stat(pid, ppid=1, start=100, name="runtime name)"):
    # start_time is field 22, or item 18 after the state field.
    return f"{pid} ({name}) S {ppid} " + "0 " * 17 + f"{start} 0\n"


def _proc(tmp_path, pid, *, ppid=1, start=100, comm="ollama", cmd=b"ollama\0serve\0", exe="/usr/bin/ollama"):
    entry = tmp_path / str(pid)
    (entry / "fd").mkdir(parents=True)
    (entry / "stat").write_text(_stat(pid, ppid, start))
    (entry / "comm").write_text(comm + "\n")
    (entry / "cmdline").write_bytes(cmd)
    os.symlink(exe, entry / "exe")
    return entry


def _net(tmp_path, name="tcp", rows=()):
    (tmp_path / "net").mkdir(exist_ok=True)
    (tmp_path / "net" / name).write_text("sl local_address rem_address st tx_queue rx_queue tr tm->when retrnsmt uid timeout inode\n" + "\n".join(rows))


def _sample(uuid="GPU-a", pid=10, memory=1.0, start=None):
    state = TelemetryFieldState("used_gpu_memory_mib", "observed")
    return pt.GPUProcessSample(uuid, pid, "runtime", memory, STAMP, "fixture", (state,), start)


def test_process_identity_validation_and_deterministic_serialization():
    process = pt.RuntimeProcessIdentity(10, 20, 1, "/x/ollama", "ollama", ("ollama", "serve"), (8081, 8081), "ollama", ("b", "a", "a"), STAMP)
    assert process.stable_identity == (10, 20)
    assert process.listening_ports == (8081,)
    assert process.discovery_sources == ("a", "b")
    with pytest.raises(ValueError, match="PID"):
        pt.RuntimeProcessIdentity(True, None, None, None, None, (), (), None, (), STAMP)
    with pytest.raises(ValueError, match="ports"):
        pt.RuntimeProcessIdentity(1, None, None, None, None, (), (0,), None, (), STAMP)


@pytest.mark.parametrize("text", [
    _stat(12, 4, 88, "name with spaces"),
    _stat(12, 4, 88, "nested (name) here"),
])
def test_parse_proc_stat_handles_spaces_and_parentheses(text):
    assert pt.parse_proc_stat(text) == (12, 4, 88)


@pytest.mark.parametrize("text", ["", "12 bad", "12 (x) S 1"])
def test_parse_proc_stat_rejects_malformed_records(text):
    with pytest.raises(ValueError, match="malformed"):
        pt.parse_proc_stat(text)


def test_discovery_uses_fixture_procfs_and_never_reads_environment(tmp_path):
    _net(tmp_path, rows=["0: 0100007F:2CAA 00000000:0000 0A 0:0 0:0 0 0 0 55"])
    _net(tmp_path, "tcp6")
    entry = _proc(tmp_path, 10)
    os.symlink("socket:[55]", entry / "fd" / "1")
    result = pt.discover_runtime_processes(proc_root=tmp_path, backend="ollama", endpoint_port=11434, timestamp_utc=STAMP)
    assert result.inspected_pid_count == 1
    assert result.processes[0].pid == 10
    assert result.processes[0].listening_ports == (11434,)


def test_discovery_never_requests_process_environment_file(tmp_path, monkeypatch):
    _net(tmp_path); _net(tmp_path, "tcp6"); _proc(tmp_path, 10)
    requested, original_open = [], __import__("pathlib").Path.open
    def tracked_open(path, *args, **kwargs):
        requested.append(path.name)
        if path.name == "environ": raise AssertionError("environment must not be read")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(__import__("pathlib").Path, "open", tracked_open)
    pt.discover_runtime_processes(proc_root=tmp_path, backend="ollama", timestamp_utc=STAMP)
    assert "environ" not in requested


def test_discovery_handles_missing_and_malformed_processes(tmp_path):
    _net(tmp_path); _net(tmp_path, "tcp6")
    broken = tmp_path / "10"; broken.mkdir(); (broken / "stat").write_text("bad")
    result = pt.discover_runtime_processes(proc_root=tmp_path, timestamp_utc=STAMP)
    assert result.processes == ()
    assert result.errors[0].operation == "proc_stat"


def test_empty_proc_cmdline_is_valid_absence_and_cannot_classify(tmp_path):
    _net(tmp_path); _net(tmp_path, "tcp6")
    _proc(tmp_path, 10, comm="worker", cmd=b"", exe="/usr/bin/worker")
    result = pt.discover_runtime_processes(proc_root=tmp_path, timestamp_utc=STAMP)
    assert result.processes == ()
    assert not any(error.operation == "proc_cmdline" for error in result.errors)


def test_empty_proc_cmdline_keeps_independent_executable_classification(tmp_path):
    _net(tmp_path); _net(tmp_path, "tcp6")
    _proc(tmp_path, 10, comm="worker", cmd=b"", exe="/usr/bin/ollama")
    result = pt.discover_runtime_processes(proc_root=tmp_path, backend="ollama", timestamp_utc=STAMP)
    assert result.processes[0].command_line == ()
    assert result.processes[0].backend_hint == "ollama"
    assert "proc_cmdline" not in result.processes[0].discovery_sources
    assert not any(error.operation == "proc_cmdline" for error in result.errors)


def test_nonempty_malformed_proc_cmdline_remains_error(tmp_path):
    _net(tmp_path); _net(tmp_path, "tcp6")
    _proc(tmp_path, 10, cmd=b"ollama\0serve", exe="/usr/bin/worker")
    result = pt.discover_runtime_processes(proc_root=tmp_path, timestamp_utc=STAMP)
    assert any(error.operation == "proc_cmdline" for error in result.errors)


def test_discovery_preserves_parent_child_and_pid_reuse_start_time(tmp_path):
    _net(tmp_path); _net(tmp_path, "tcp6")
    _proc(tmp_path, 10, start=100, cmd=b"ollama\0serve\0")
    _proc(tmp_path, 11, ppid=10, start=101, comm="ollama runner", cmd=b"ollama\0runner\0")
    found = pt.discover_runtime_processes(proc_root=tmp_path, backend="ollama", timestamp_utc=STAMP).processes
    assert [(item.pid, item.parent_pid, item.start_time_ticks) for item in found] == [(10, 1, 100), (11, 10, 101)]
    assert found[0].stable_identity != pt.RuntimeProcessIdentity(10, 102, 1, None, None, (), (), "ollama", (), STAMP).stable_identity


def test_tcp_parser_filters_nonlisteners_and_orders_ipv4_ipv6():
    rows = "0: 0100007F:1F91 00000000:0000 0A 0:0 0:0 0 0 0 44\n1: 00000000:2382 00000000:0000 01 0:0 0:0 0 0 0 45"
    assert pt.parse_proc_net_tcp("header\n" + rows) == ((8081, "44"),)


@pytest.mark.parametrize("row", [
    "0: 0100007F:1F91 00000000:0000 ZZ 0:0 0:0 0 0 0 44",
    "0: :1F91 00000000:0000 0A 0:0 0:0 0 0 0 44",
    "0: 0100007F:0000 00000000:0000 0A 0:0 0:0 0 0 0 44",
    "0: 0100007F:1F91 00000000:0000 0A 0:0 0:0 0 0 0 nope",
])
def test_tcp_parser_reports_malformed_socket_rows(row):
    records, errors = pt.parse_proc_net_tcp("header\n" + row, with_errors=True)
    assert records == () and errors[0].operation == "proc_net"


def test_tcp_parser_ignores_valid_non_listening_row():
    records, errors = pt.parse_proc_net_tcp("header\n0: 0100007F:1F91 00000000:0000 01 0:0 0:0 0 0 0 44", with_errors=True)
    assert records == () and errors == ()


def test_socket_ownership_supports_multiple_owners_and_ipv6(tmp_path):
    _net(tmp_path, rows=["0: 0100007F:1F91 00000000:0000 0A 0:0 0:0 0 0 0 44"])
    _net(tmp_path, "tcp6", ["0: 00000000000000000000000000000000:2CAA 00000000000000000000000000000000:0000 0A 0:0 0:0 0 0 0 55"])
    for pid in (10, 11):
        entry = _proc(tmp_path, pid, comm="llama-server", cmd=b"llama-server\0--port\08081\0", exe="/build/llama-server")
        os.symlink("socket:[44]", entry / "fd" / "1")
        os.symlink("socket:[55]", entry / "fd" / "2")
    discovery = pt.discover_runtime_processes(proc_root=tmp_path, backend="llama_cpp", timestamp_utc=STAMP)
    found = discovery.processes
    assert [item.listening_ports for item in found] == [(8081, 11434), (8081, 11434)]


def test_classification_does_not_accept_misleading_substrings(tmp_path):
    _net(tmp_path); _net(tmp_path, "tcp6")
    _proc(tmp_path, 10, comm="python", cmd=b"python\0script.py\0--ollama-not-a-server\0", exe="/usr/bin/python")
    assert pt.discover_runtime_processes(proc_root=tmp_path, timestamp_utc=STAMP).processes == ()


def test_gpu_process_parser_handles_empty_quoted_many_and_markers():
    empty = pt.parse_nvidia_gpu_process_csv("\n", timestamp_utc=STAMP)
    assert empty.successful and empty.samples == ()
    result = pt.parse_nvidia_gpu_process_csv('10,"name, with comma",GPU-b,0\n11,other,GPU-a,N/A\n', timestamp_utc=STAMP)
    assert [(item.gpu_uuid, item.pid, item.used_gpu_memory_mib) for item in result.samples] == [("GPU-a", 11, None), ("GPU-b", 10, 0.0)]
    assert result.samples[0].field_states[0].state == "unavailable"


@pytest.mark.parametrize("raw,operation", [
    ("10,x,GPU-a\n", "csv_schema"),
    ("10,x,GPU-a,1,extra\n", "csv_schema"),
    ("bad,x,GPU-a,1\n", "csv_pid"),
    ("10,x,N/A,1\n", "csv_uuid"),
])
def test_gpu_process_parser_rejects_schema_and_identity_errors(raw, operation):
    result = pt.parse_nvidia_gpu_process_csv(raw, timestamp_utc=STAMP)
    assert result.samples == ()
    assert result.errors[0].operation == operation


def test_gpu_process_duplicate_pairs_are_excluded_as_groups():
    first = pt.parse_nvidia_gpu_process_csv("10,x,GPU-a,1\n10,y,GPU-a,2\n11,z,GPU-b,3\n", timestamp_utc=STAMP)
    second = pt.parse_nvidia_gpu_process_csv("11,z,GPU-b,3\n10,y,GPU-a,2\n10,x,GPU-a,1\n", timestamp_utc=STAMP)
    assert [item.gpu_uuid for item in first.samples] == ["GPU-b"]
    assert first.to_dict() == second.to_dict()


@pytest.mark.parametrize("memory", ["-1", "nan", "inf", "bad"])
def test_gpu_process_invalid_memory_is_row_evidence(memory):
    result = pt.parse_nvidia_gpu_process_csv(f"10,x,GPU-a,{memory}\n", timestamp_utc=STAMP)
    assert result.samples[0].used_gpu_memory_mib is None
    assert result.samples[0].field_states[0].state == "malformed"


class _Runner:
    def __init__(self, result): self.result, self.calls = result, []
    def __call__(self, *args): self.calls.append(args); return self.result


@pytest.mark.parametrize("result", [
    CommandResult(None, executable_missing=True), CommandResult(1, stderr="bad"), CommandResult(None, timed_out=True),
    CommandResult(0, stdout_overflow=True), CommandResult(0, stderr_overflow=True),
])
def test_gpu_process_collection_failures_are_bounded(result):
    runner = _Runner(result)
    collected = pt.collect_nvidia_gpu_processes(runner=runner, timestamp_utc=STAMP)
    assert not collected.successful and collected.samples == ()
    assert runner.calls[0][0] == pt.nvidia_process_command()


def test_gpu_process_collection_exact_vector_and_no_shell_contract():
    runner = _Runner(CommandResult(0, stdout="10,x,GPU-a,1\n"))
    collected = pt.collect_nvidia_gpu_processes(runner=runner, timestamp_utc=STAMP)
    assert collected.successful and collected.samples[0].pid == 10
    assert runner.calls[0][1:] == (pt.NVIDIA_PROCESS_TIMEOUT_SECONDS, pt.MAX_NVIDIA_PROCESS_STDOUT_BYTES, pt.MAX_NVIDIA_PROCESS_STDERR_BYTES)


def test_default_process_runner_uses_an_argument_vector_without_shell(monkeypatch):
    captured = {}
    class Process:
        stdout = __import__("io").BytesIO(b"")
        stderr = __import__("io").BytesIO(b"")
        def wait(self, timeout=None): return 0
        def kill(self): raise AssertionError("unexpected kill")
    def popen(command, **kwargs):
        captured["command"], captured["kwargs"] = command, kwargs
        return Process()
    monkeypatch.setattr(pt.subprocess, "Popen", popen)
    assert pt._bounded_subprocess(pt.nvidia_process_command(), 1, 16, 16).returncode == 0
    assert captured["command"] == pt.nvidia_process_command()
    assert captured["kwargs"]["shell"] is False


def test_attribution_confidence_and_declared_observed_distinctions():
    confirmed = pt.RuntimeProcessIdentity(10, 100, 1, "/usr/bin/ollama", "ollama", ("ollama", "serve"), (11434,), "ollama", ("proc_stat", "proc_fd_socket"), STAMP)
    probable = pt.RuntimeProcessIdentity(11, None, 1, "/build/llama-server", "llama-server", ("llama-server",), (), "llama_cpp", ("proc_cmdline",), STAMP)
    samples = (_sample("GPU-confirmed", 10, start=100), _sample("GPU-probable", 11), _sample("GPU-other", 99))
    result = pt.attribute_runtime_gpus(backend="ollama", endpoint_port=11434, processes=(confirmed, probable), gpu_process_samples=samples,
                                       declared_gpu_uuids=("GPU-confirmed", "GPU-declared"), timestamp_utc=STAMP)
    values = {item.gpu_uuid: item for item in result.attributions}
    assert values["GPU-confirmed"].confidence == "confirmed"
    assert values["GPU-declared"].confidence == "profile-declared only"
    assert "GPU-other" not in values
    assert any(error.operation == "runtime_match" for error in result.errors)
    assert result.declared_gpu_uuids == ("GPU-confirmed", "GPU-declared")
    assert result.observed_gpu_uuids == ("GPU-confirmed",)


def test_attribution_probable_stale_pid_and_no_cuda_ordinal_conversion():
    process = pt.RuntimeProcessIdentity(10, None, 1, "/build/llama-server", "llama-server", ("llama-server", "-dev", "CUDA0,CUDA1"), (), "llama_cpp", ("proc_cmdline",), STAMP)
    result = pt.attribute_runtime_gpus(backend="llama_cpp", endpoint_port=8081, processes=(process,), gpu_process_samples=(_sample("GPU-a", 10),), declared_gpu_uuids=("GPU-declared",), timestamp_utc=STAMP)
    assert [item.confidence for item in result.attributions] == ["probable", "profile-declared only"]
    assert "CUDA0" not in result.observed_gpu_uuids


def test_attribution_excludes_ambiguous_pid_and_unrelated_declared_gpu():
    first = pt.RuntimeProcessIdentity(10, 100, 1, "/x/ollama", "ollama", ("ollama",), (11434,), "ollama", ("proc_stat",), STAMP)
    reused = pt.RuntimeProcessIdentity(10, 101, 1, "/x/ollama", "ollama", ("ollama",), (11434,), "ollama", ("proc_stat",), STAMP)
    result = pt.attribute_runtime_gpus(backend="ollama", endpoint_port=11434, processes=(reused, first),
                                       gpu_process_samples=(_sample("GPU-declared", 99), _sample("GPU-other", 10, start=100)),
                                       declared_gpu_uuids=("GPU-declared",), timestamp_utc=STAMP)
    assert result.observed_gpu_uuids == ()
    assert result.attributions[0].confidence == "profile-declared only"
    assert result.attributions[0].runtime_pid is None
    assert {error.operation for error in result.errors} == {"endpoint_owner", "runtime_match", "runtime_pid"}


def test_multiple_endpoint_owners_are_probable_not_confirmed_and_reorder_stable():
    processes = tuple(pt.RuntimeProcessIdentity(pid, 100 + pid, 1, "/x/ollama", "ollama", ("ollama",), (11434,), "ollama", ("proc_stat",), STAMP) for pid in (10, 11))
    samples = (_sample("GPU-a", 10, start=110),)
    first = pt.attribute_runtime_gpus(backend="ollama", endpoint_port=11434, processes=processes, gpu_process_samples=samples, timestamp_utc=STAMP)
    second = pt.attribute_runtime_gpus(backend="ollama", endpoint_port=11434, processes=reversed(processes), gpu_process_samples=samples, timestamp_utc=STAMP)
    assert first.attributions[0].confidence == "probable"
    assert first.to_dict() == second.to_dict()


def test_start_mismatch_is_not_runtime_observed_and_reconciliation_is_safe():
    process = pt.RuntimeProcessIdentity(10, 100, 1, "/x/ollama", "ollama", ("ollama",), (11434,), "ollama", ("proc_stat",), STAMP)
    mismatch = pt.attribute_runtime_gpus(backend="ollama", endpoint_port=11434, processes=(process,),
                                         gpu_process_samples=(_sample("GPU-declared", 10, start=101),),
                                         declared_gpu_uuids=("GPU-declared",), timestamp_utc=STAMP)
    assert mismatch.observed_gpu_uuids == ()
    assert mismatch.attributions[0].confidence == "profile-declared only"
    assert any(error.operation == "runtime_pid_reuse" for error in mismatch.errors)
    stable, errors = pt.reconcile_gpu_process_stable_identities((_sample("GPU-a", 10),), before=(process,), after=(process,))
    assert stable[0].process_start_time_ticks == 100 and not errors
    changed, errors = pt.reconcile_gpu_process_stable_identities((_sample("GPU-a", 10),), before=(process,),
                                                                   after=(pt.RuntimeProcessIdentity(10, 101, 1, "/x/ollama", "ollama", ("ollama",), (), "ollama", (), STAMP),))
    assert changed[0].process_start_time_ticks is None
    assert errors[0].operation == "runtime_pid_reuse"


def test_reconciliation_rejects_conflicting_metadata_and_clears_stale_start():
    first = pt.RuntimeProcessIdentity(10, 100, 1, "/x/ollama", "ollama", ("ollama",), (), "ollama", (), STAMP)
    conflict = pt.RuntimeProcessIdentity(10, 100, 1, "/x/other", "ollama", ("ollama",), (), "ollama", (), STAMP)
    stale = _sample("GPU-a", 10, start=100)
    samples, errors = pt.reconcile_gpu_process_stable_identities((stale,), before=(first, conflict), after=(first,))
    assert samples[0].process_start_time_ticks is None
    assert any(error.operation == "runtime_identity" for error in errors)
    stable, errors = pt.reconcile_gpu_process_stable_identities((stale,), before=(first, first), after=(first, first))
    assert stable[0].process_start_time_ticks == 100 and not errors


def test_process_result_rejects_ambiguous_direct_identity_sets():
    first = pt.RuntimeProcessIdentity(10, 100, 1, None, None, (), (), "ollama", (), STAMP)
    conflict = pt.RuntimeProcessIdentity(10, 100, 1, "/x/ollama", None, (), (), "ollama", (), STAMP)
    with pytest.raises(ValueError, match="conflicting"):
        pt.ProcessDiscoveryResult((first, conflict), (), STAMP)
    with pytest.raises(ValueError, match="multiple stable"):
        pt.RuntimeAttributionResult((first, pt.RuntimeProcessIdentity(10, 101, 1, None, None, (), (), "ollama", (), STAMP)), (), (), (), (), STAMP)


def test_incomplete_socket_evidence_prevents_confirmation():
    process = pt.RuntimeProcessIdentity(10, 100, 1, "/x/ollama", "ollama", ("ollama",), (11434,), "ollama", ("proc_stat",), STAMP)
    result = pt.attribute_runtime_gpus(backend="ollama", endpoint_port=11434, processes=(process,),
                                       gpu_process_samples=(_sample("GPU-a", 10, start=100),), timestamp_utc=STAMP,
                                       socket_evidence_complete=False)
    assert result.attributions[0].confidence == "probable"


def test_direct_result_invariants_reject_conflicting_query_and_relationships():
    with pytest.raises(ValueError, match="fixed NVIDIA"):
        pt.GPUProcessCollectionResult((), (), STAMP, ("other",), True)
    with pytest.raises(ValueError, match="successful"):
        pt.GPUProcessCollectionResult((), (), STAMP, pt.nvidia_process_command(), "yes")
    first = pt.RuntimeProcessIdentity(10, None, 1, None, None, (), (), "ollama", (), STAMP)
    second = pt.RuntimeProcessIdentity(10, 100, 1, None, None, (), (), "ollama", (), STAMP)
    result = pt.attribute_runtime_gpus(backend="ollama", endpoint_port=None, processes=(first, second), gpu_process_samples=(), timestamp_utc=STAMP)
    assert any(error.operation == "runtime_pid" for error in result.errors)


def test_endpoint_owner_filters_exact_requested_port_before_ambiguity():
    owner = pt.RuntimeProcessIdentity(10, 100, 1, "/x/ollama", "ollama", ("ollama", "serve"), (11434,), "ollama", ("proc_stat",), STAMP)
    other = pt.RuntimeProcessIdentity(11, 101, 1, "/x/ollama", "ollama", ("ollama", "serve"), (11435,), "ollama", ("proc_stat",), STAMP)
    result = pt.attribute_runtime_gpus(backend="ollama", endpoint_port=11434, processes=(owner, other), gpu_process_samples=(_sample("GPU-a", 10, start=100),), timestamp_utc=STAMP, socket_evidence_complete=True)
    assert result.attributions[0].confidence == "confirmed"
    assert not any(error.operation == "endpoint_owner" for error in result.errors)


def test_ollama_worker_requires_proven_lineage_and_is_probable_when_socket_partial():
    owner = pt.RuntimeProcessIdentity(10, 100, 1, "/x/ollama", "ollama", ("ollama", "serve"), (11434,), "ollama", ("proc_stat",), STAMP)
    worker = pt.RuntimeProcessIdentity(20, 200, 10, "/usr/local/lib/ollama/llama-server", "llama-server", ("llama-server",), (), None, ("proc_stat", "proc_exe"), STAMP)
    result = pt.attribute_runtime_gpus(backend="ollama", endpoint_port=11434, processes=(owner, worker), gpu_process_samples=(_sample("GPU-a", 20, start=200),), timestamp_utc=STAMP, socket_evidence_complete=False)
    assert result.attributions[0].confidence == "probable"
    assert "ollama_worker_lineage" in result.attributions[0].evidence_sources
    standalone = pt.RuntimeProcessIdentity(21, 201, 1, "/usr/local/lib/ollama/llama-server", "llama-server", ("llama-server",), (), None, ("proc_stat",), STAMP)
    blocked = pt.attribute_runtime_gpus(backend="ollama", endpoint_port=11434, processes=(owner, standalone), gpu_process_samples=(_sample("GPU-b", 21, start=201),), timestamp_utc=STAMP)
    assert blocked.attributions == ()
    assert any(error.operation == "runtime_lineage" for error in blocked.errors)


def test_timestamp_only_duplicate_identity_deduplicates_to_later_observation():
    early = pt.RuntimeProcessIdentity(10, 100, 1, "/x/ollama", "ollama", ("ollama", "serve"), (11434,), "ollama", ("proc_stat",), "2026-08-01T12:00:00Z")
    late = pt.RuntimeProcessIdentity(10, 100, 1, "/x/ollama", "ollama", ("ollama", "serve"), (11434,), "ollama", ("proc_stat",), "2026-08-01T12:01:00Z")
    result = pt.attribute_runtime_gpus(backend="ollama", endpoint_port=11434, processes=(early, late), gpu_process_samples=(), timestamp_utc=STAMP)
    assert result.processes == (late,)
    assert not any(error.operation == "runtime_identity" for error in result.errors)


@pytest.mark.parametrize("field, value", [
    ("executable", "/x/other"), ("command_line", ("ollama", "other")),
    ("parent_pid", 2), ("listening_ports", (11435,)),
])
def test_substantive_identity_conflicts_remain_ambiguous(field, value):
    base = pt.RuntimeProcessIdentity(10, 100, 1, "/x/ollama", "ollama", ("ollama", "serve"), (11434,), "ollama", ("proc_stat",), STAMP)
    values = base.to_dict(); values[field] = value; values["timestamp_utc"] = "2026-08-01T12:01:00Z"
    conflict = pt.RuntimeProcessIdentity(**values)
    result = pt.attribute_runtime_gpus(backend="ollama", endpoint_port=11434, processes=(base, conflict), gpu_process_samples=(), timestamp_utc=STAMP)
    assert any(error.operation == "runtime_identity" for error in result.errors)
