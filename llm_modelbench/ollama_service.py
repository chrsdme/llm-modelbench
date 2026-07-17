"""Human-supervised systemd control for temporary Ollama KV-cache changes.

The controller is deliberately narrow:

* it discovers the systemd unit that owns the process listening on the
  configured Ollama port instead of assuming ``ollama.service``;
* it writes only one dedicated drop-in containing
  ``OLLAMA_KV_CACHE_TYPE``;
* it verifies systemd's merged environment before restart and the live
  process environment after restart;
* it never reads or stores a sudo password; the normal ``sudo`` program owns
  the controlling-terminal prompt;
* it refuses to restart a unit whose CUDA UUID binding no longer exists.
"""
from __future__ import annotations

import hashlib
import os
import re
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

_ALLOWED_KV = {"q8_0", "q4_0"}
_UNIT_RE = re.compile(r"^[A-Za-z0-9_.@-]+(?:\.service)?$")
_MARKER = "# Managed temporarily by llmb repair; safe to remove after restoration."
_SS_PID_RE = re.compile(r"pid=(\d+)")
_KV_ENV_RE = re.compile(r'OLLAMA_KV_CACHE_TYPE=([^\s"\']+)')
_CUDA_ENV_RE = re.compile(r'CUDA_VISIBLE_DEVICES=([^\s"\']+)')


class ServiceControlError(RuntimeError):
    """Raised when a supervised service operation cannot be completed safely."""


@dataclass(frozen=True)
class ActiveService:
    unit: str
    pid: int
    port: int


@dataclass
class DropInSnapshot:
    existed: bool
    content: Optional[bytes] = None
    sha256: Optional[str] = None


@dataclass
class ServicePhaseResult:
    phase: str
    unit: str
    kv_type: Optional[str]
    active: bool
    verified: bool
    observed_kv_type: Optional[str]
    note: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "unit": self.unit,
            "kv_type": self.kv_type,
            "active": self.active,
            "verified": self.verified,
            "observed_kv_type": self.observed_kv_type,
            "note": self.note,
        }


def _run_checked(
    run: Callable[..., subprocess.CompletedProcess],
    argv: Sequence[str],
    *,
    timeout: int = 15,
) -> subprocess.CompletedProcess:
    try:
        result = run(
            list(argv),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ServiceControlError(f"required command not found: {argv[0]!r}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ServiceControlError(f"command timed out: {' '.join(argv)}") from exc
    if result.returncode != 0:
        stderr = str(getattr(result, "stderr", "") or "").strip()
        raise ServiceControlError(
            f"command failed ({result.returncode}): {' '.join(argv)}"
            + (f"\n{stderr[:1000]}" if stderr else "")
        )
    return result


def _resolve_executable(command: str) -> str:
    """Resolve *command* without ``shutil.which``.

    Some tests monkeypatch ``shutil.which`` through the shared stdlib module
    object. Using it here made an unrelated repair test silently resolve
    ``ss`` to ``systemctl``. Search PATH directly so the privileged command
    is deterministic and isolated from those mocks. An explicit path is
    preserved unchanged.
    """
    if os.path.sep in command:
        return command
    for directory in os.get_exec_path():
        candidate = Path(directory) / command
        try:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
        except OSError:
            continue
    return command


def _listener_pid_from_ss(output: str, port: int) -> Optional[int]:
    """Return the PID bound to *local* TCP port ``port`` from ``ss`` output."""
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        # Typical columns: State Recv-Q Send-Q Local:Port Peer:Port Process
        local_address = fields[3]
        if not (local_address.endswith(f":{port}") or local_address.endswith(f"]:{port}")):
            continue
        match = _SS_PID_RE.search(line)
        if match:
            return int(match.group(1))
    return None


def discover_active_service(
    *,
    port: int = 11434,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    sudo_command: str = "sudo",
    systemctl_command: str = "systemctl",
    candidate_pattern: str = "ollama",
    warn_fn: Optional[Callable[[str], None]] = None,
    use_sudo: bool = True,
    ss_command: str = "ss",
) -> ActiveService:
    """Find the systemd unit owning the PID listening on the Ollama port.

    ``use_sudo=False`` is for read-only, non-privileged contexts (e.g. plan
    preview / dry-run) that must never trigger a password prompt just to
    describe current state. Without sudo, ``ss`` may not reveal ownership of
    a socket held by a different user, in which case this raises the same
    "no process with visible ownership found" error as a real absence would
    -- callers in that context should treat it as "unknown", not escalate.

    Discovery is evidence-based: the listener PID from ``ss`` must match a
    candidate unit's ``MainPID``. No service is selected by name convention.
    The caller is responsible for obtaining sudo authentication first when
    process ownership details require it.

    If another ollama-pattern unit is *simultaneously alive* (has its own
    live MainPID) alongside the winner, that's a real resource/port
    contention risk distinct from one merely dormant unit (e.g. a
    crash-looping ``ollama.service`` with ``MainPID=0``, which is not a
    conflict and does not trigger this). Reported via ``warn_fn`` rather
    than silently proceeding; does not change which unit is selected.
    """
    ss_argv = (
        ([sudo_command] if (use_sudo and os.geteuid() != 0) else []) + [ss_command, "-H", "-tlnp"]
    )
    ss_result = _run_checked(run, ss_argv, timeout=15)
    listening_pid = _listener_pid_from_ss(str(ss_result.stdout or ""), int(port))
    if listening_pid is None:
        raise ServiceControlError(
            f"no process with visible ownership found listening on TCP port {port}; "
            "is Ollama running, and was sudo authentication successful?"
        )

    list_result = _run_checked(
        run,
        [systemctl_command, "list-units", "--type=service", "--all", "--no-legend", "--plain"],
        timeout=15,
    )
    candidates: List[str] = []
    for line in str(list_result.stdout or "").splitlines():
        stripped = line.strip().lstrip("●").strip()
        if not stripped:
            continue
        unit_name = stripped.split()[0]
        if candidate_pattern.lower() in unit_name.lower() and unit_name.endswith(".service"):
            candidates.append(unit_name)

    main_pids: Dict[str, Optional[int]] = {}
    for unit_name in candidates:
        show_result = _run_checked(
            run,
            [systemctl_command, "show", unit_name, "--property=MainPID", "--value"],
            timeout=10,
        )
        pid_text = str(show_result.stdout or "").strip()
        main_pids[unit_name] = int(pid_text) if pid_text.isdigit() and int(pid_text) > 0 else None

    matches = [name for name, pid in main_pids.items() if pid == listening_pid]

    if len(matches) > 1:
        raise ServiceControlError(
            f"process {listening_pid} on port {port} matches multiple systemd units: {matches}; "
            "refusing an ambiguous service restart"
        )
    if len(matches) == 1:
        winner = matches[0]
        if warn_fn:
            conflicts = [
                name for name, pid in main_pids.items()
                if name != winner and pid is not None
            ]
            if conflicts:
                warn_fn(
                    f"{winner} owns port {port} (PID {listening_pid}), but other "
                    f"ollama-pattern units are also simultaneously alive: {conflicts}. "
                    f"This is a resource/port contention risk worth investigating "
                    f"even though {winner} is being used."
                )
        return ActiveService(winner, listening_pid, int(port))
    raise ServiceControlError(
        f"process {listening_pid} is listening on port {port}, but no unit matching "
        f"{candidate_pattern!r} has that MainPID; refusing to guess. Checked: "
        f"{candidates or '(no candidate units found)'}"
    )


def discover_active_unit(**kwargs: Any) -> str:
    """Compatibility wrapper returning only the discovered unit name."""
    return discover_active_service(**kwargs).unit


class OllamaServiceController:
    """Manage a temporary KV-cache drop-in with explicit human supervision."""

    def __init__(
        self,
        unit: str = "ollama.service",
        *,
        port: int = 11434,
        run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        input_fn: Callable[[str], str] = input,
        isatty_fn: Callable[[], bool] = lambda: bool(os.isatty(0) and os.isatty(1)),
        sleep_fn: Callable[[float], None] = time.sleep,
        sudo_command: str = "sudo",
        systemctl_command: str = "systemctl",
        nvidia_smi_command: str = "nvidia-smi",
        force_password_prompt: bool = True,
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        auto_confirm: bool = False,
        kv_read_helper_path: Path = Path("/usr/local/libexec/llmb-read-kv-env.sh"),
        kv_read_helper_exists_fn: Callable[[Path], bool] = lambda p: p.exists(),
        ss_command: Optional[str] = None,
    ) -> None:
        if not _UNIT_RE.fullmatch(unit):
            raise ValueError(f"unsafe systemd unit name: {unit!r}")
        if int(port) < 1 or int(port) > 65535:
            raise ValueError(f"invalid Ollama TCP port: {port!r}")
        self.kv_read_helper_path = kv_read_helper_path
        self._kv_read_helper_exists_fn = kv_read_helper_exists_fn
        self.unit = unit if unit.endswith(".service") else f"{unit}.service"
        self.port = int(port)
        self._run_impl = run
        self._input = input_fn
        self._isatty = isatty_fn
        self._sleep = sleep_fn
        self.sudo_command = sudo_command
        self.systemctl_command = systemctl_command
        self.nvidia_smi_command = nvidia_smi_command
        # Auto-confirm must exercise the exact executable path authorized in
        # sudoers. Resolve it without shutil.which to avoid shared-module test
        # monkeypatch pollution. Interactive mode keeps the historical token.
        self.ss_command = ss_command or (_resolve_executable("ss") if auto_confirm else "ss")
        self.force_password_prompt = bool(force_password_prompt)
        self.event_callback = event_callback
        self.auto_confirm = bool(auto_confirm)
        self.dropin_dir = Path("/etc/systemd/system") / f"{self.unit}.d"
        # Sort after ordinary override.conf/99-*.conf files. The merged-
        # environment verification below remains authoritative if an even
        # later unmanaged file exists.
        self.dropin_path = self.dropin_dir / "zzzz-llmb-repair-kv.conf"
        self.snapshot: Optional[DropInSnapshot] = None
        self.mutation_started = False
        self.discovery_warnings: list[str] = []
        self.events: list[Dict[str, Any]] = []
        self._last_restart_error: Optional[str] = None

    @classmethod
    def for_active_service(cls, *, warn_fn: Optional[Callable[[str], None]] = None, **kwargs: Any) -> "OllamaServiceController":
        run = kwargs.get("run", subprocess.run)
        sudo_command = kwargs.get("sudo_command", "sudo")
        systemctl_command = kwargs.get("systemctl_command", "systemctl")
        port = int(kwargs.get("port", 11434))
        auto_confirm = bool(kwargs.get("auto_confirm", False))
        ss_command = kwargs.get("ss_command") or (_resolve_executable("ss") if auto_confirm else "ss")
        kwargs["ss_command"] = ss_command
        collected: List[str] = []
        active = discover_active_service(
            port=port,
            run=run,
            sudo_command=sudo_command,
            systemctl_command=systemctl_command,
            warn_fn=collected.append,
            ss_command=ss_command,
        )
        controller = cls(active.unit, **kwargs)
        controller.discovery_warnings = collected
        for message in collected:
            if warn_fn:
                warn_fn(message)
        return controller

    @property
    def privileged_prefix(self) -> list[str]:
        if os.geteuid() == 0:
            return []
        if self.auto_confirm:
            # -n: never prompt. If the NOPASSWD sudoers rule isn't actually
            # in place, this fails the command immediately and visibly
            # instead of hanging on a password prompt with no TTY attached,
            # or (worse) blocking a supposedly-unattended run indefinitely.
            return [self.sudo_command, "-n"]
        return [self.sudo_command]

    def verify_noninteractive_sudo_ready(self) -> None:
        """Preflight check for --auto-confirm: prove passwordless sudo
        actually works before starting DISCOVER, rather than discovering the
        gap partway through a cascade. Must not silently fall back to
        interactive mode -- a misconfigured unattended run should fail
        immediately and deterministically, not hang or prompt.

        Deliberately does NOT use ``sudo -n -v``: a scoped, per-command
        NOPASSWD rule (the only kind this project recommends) grants no
        general credential validation at all, so ``-v`` would fail even when
        every command this controller actually runs would succeed
        passwordlessly. Instead this probes one real, side-effect-free
        command already in the allowed set (``ss -H -tlnp``), so a pass here
        means the actual rule genuinely works, not just that *some* sudo
        access exists.
        """
        if not self.auto_confirm or os.geteuid() == 0:
            return
        # Invalidate any cached sudo timestamp first. Without this, a recent
        # interactive sudo command can make the probe pass even when the
        # NOPASSWD rule is missing or has the wrong executable path. ``-k``
        # never prompts; the following ``-n`` command therefore proves the
        # unattended rule itself.
        self._run([self.sudo_command, "-k"], check=False, capture_output=False, timeout=10)
        command = [self.ss_command, "-H", "-tlnp"]
        result = self._run(command, privileged=True, check=False, timeout=10)
        if result.returncode != 0:
            stderr = str(getattr(result, "stderr", "") or "").strip()
            rendered = shlex.join([self.sudo_command, "-n", *command])
            raise ServiceControlError(
                "--auto-confirm requires a scoped NOPASSWD sudoers rule covering the exact "
                f"commands this controller runs, but {rendered!r} failed (exit "
                f"{result.returncode})"
                + (f": {stderr[:1000]}" if stderr else "")
                + ". Install or correct the sudoers rule; refusing to fall back to an "
                "interactive prompt in an unattended run."
            )

    def _run(
        self,
        command: Sequence[str],
        *,
        privileged: bool = False,
        check: bool = True,
        capture_output: bool = True,
        text: bool = True,
        timeout: int = 30,
    ) -> subprocess.CompletedProcess:
        argv = list(self.privileged_prefix if privileged else []) + list(command)
        try:
            result = self._run_impl(
                argv,
                check=False,
                capture_output=capture_output,
                text=text,
                timeout=timeout,
            )
        except FileNotFoundError as exc:
            raise ServiceControlError(f"required command not found: {argv[0]!r}") from exc
        except subprocess.TimeoutExpired as exc:
            raise ServiceControlError(f"command timed out: {' '.join(argv)}") from exc
        if check and result.returncode != 0:
            stderr = str(getattr(result, "stderr", "") or "").strip()
            raise ServiceControlError(
                f"command failed ({result.returncode}): {' '.join(argv)}"
                + (f"\n{stderr[:1000]}" if stderr else "")
            )
        return result

    def require_supervised_tty(self) -> None:
        if os.geteuid() != 0 and not self._isatty():
            raise ServiceControlError(
                "automatic Ollama service restarts require a real interactive terminal so sudo can ask for the password"
            )

    def confirm(self, phase: str, message: str, *, keyword: str = "RESTART") -> None:
        if self.auto_confirm:
            # Loud on purpose: auto-confirm removes the human checkpoint, so
            # every phase it skipped still needs to be visible in the log,
            # not silently invisible. This does not replace typed
            # confirmation with nothing -- it replaces it with an explicit,
            # permanent record that the phase ran unattended.
            print(
                f"\nPRIVILEGED OLLAMA SERVICE PHASE: {phase} (--auto-confirm: "
                f"proceeding without a typed {keyword})\n{message}"
            )
            return
        self.require_supervised_tty()
        answer = self._input(
            f"\nPRIVILEGED OLLAMA SERVICE PHASE: {phase}\n{message}\n"
            f"Type {keyword} to continue, or anything else to stop: "
        )
        if answer.strip() != keyword:
            raise ServiceControlError(f"operator declined service phase {phase!r}")

    def authorise_sudo(self) -> None:
        if os.geteuid() == 0:
            return
        if self.auto_confirm:
            # Relies on a scoped NOPASSWD sudoers rule for the exact commands
            # this controller runs; a blanket `sudo -v` credential refresh
            # would still prompt for a password under a properly *scoped*
            # NOPASSWD rule (which authorizes specific commands, not general
            # sudo access), so skip it and let each privileged call
            # authenticate on its own via that rule.
            return
        if self.force_password_prompt:
            self._run([self.sudo_command, "-k"], check=False, capture_output=False)
        self._run([self.sudo_command, "-v"], capture_output=False, timeout=120)

    @staticmethod
    def _is_transient_ownership_error(message: str) -> bool:
        """Same classification RC8 already uses post-restart: a missing
        MainPID or absent listener is a normal startup transition and worth
        a brief retry. A visible-but-wrong listener or unit mismatch is a
        real conflict and must never be retried away."""
        return (
            "has no running MainPID" in message
            or "no process with visible ownership found listening" in message
        )

    def _verify_owns_live_process_once(self) -> ActiveService:
        """Single-attempt ownership check, no retry. Used by the post-restart
        polling loop in ``_restart_and_wait``, which already has its own
        longer-running retry deadline; retrying here too would just nest two
        retry loops and multiply the wait unnecessarily."""
        pid = self._main_pid()
        if pid is None:
            raise ServiceControlError(
                f"{self.unit} has no running MainPID; it is not the active Ollama service"
            )
        try:
            active = discover_active_service(
                port=self.port,
                run=self._run_impl,
                sudo_command=self.sudo_command,
                systemctl_command=self.systemctl_command,
                ss_command=self.ss_command,
            )
        except ServiceControlError as exc:
            raise ServiceControlError(
                f"{self.unit} does not own the live Ollama process on port {self.port}, "
                f"or ownership could not be proven: {exc}"
            ) from exc
        if active.unit != self.unit or active.pid != pid:
            raise ServiceControlError(
                f"{self.unit} does not own the live Ollama process on port {self.port}; "
                f"the active unit is {active.unit} (PID {active.pid}). Refusing to proceed "
                "against the wrong unit."
            )
        return active

    def verify_owns_live_process(
        self, *, timeout_seconds: float = 3.0, retry: bool = True
    ) -> ActiveService:
        """Refuse a unit that does not own the configured endpoint.

        RC8 added retry tolerance for the transient window right after a
        restart, but every *other* call site of this check (the explicit
        post-discovery verification in the CLI, and the pre-mutation
        re-check at the top of ``set_kv_type``) still failed on a single
        momentary blip with no restart anywhere nearby. This applies the
        exact same transient/hard-conflict classification everywhere the
        check is used, not just after a restart: a brief, genuinely
        transient absence gets a short retry window; a real ownership
        conflict still fails immediately, every time, with no delay.
        """
        if not retry:
            return self._verify_owns_live_process_once()
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        while True:
            try:
                return self._verify_owns_live_process_once()
            except ServiceControlError as exc:
                if not self._is_transient_ownership_error(str(exc)):
                    raise
                if time.monotonic() >= deadline:
                    raise
                self._sleep(0.25)

    def _sudo_file_exists(self, path: Path) -> bool:
        result = self._run(["test", "-e", str(path)], privileged=True, check=False)
        return result.returncode == 0

    def _sudo_read_bytes(self, path: Path) -> bytes:
        result = self._run(["cat", str(path)], privileged=True, text=False)
        return bytes(result.stdout or b"")

    def snapshot_dropin(self) -> DropInSnapshot:
        if self.snapshot is not None:
            return self.snapshot
        existed = self._sudo_file_exists(self.dropin_path)
        if not existed:
            self.snapshot = DropInSnapshot(existed=False)
            return self.snapshot
        content = self._sudo_read_bytes(self.dropin_path)
        if _MARKER.encode() not in content:
            raise ServiceControlError(
                f"refusing to overwrite pre-existing unmanaged drop-in {self.dropin_path}; move or review it manually"
            )
        self.snapshot = DropInSnapshot(
            existed=True,
            content=content,
            sha256=hashlib.sha256(content).hexdigest(),
        )
        return self.snapshot

    @staticmethod
    def dropin_content(kv_type: str) -> bytes:
        if kv_type not in _ALLOWED_KV:
            raise ValueError(f"unsupported managed KV type: {kv_type!r}")
        return (
            f"{_MARKER}\n"
            "[Service]\n"
            f'Environment="OLLAMA_KV_CACHE_TYPE={kv_type}"\n'
        ).encode()

    def _install_bytes(self, content: bytes) -> None:
        # A fixed, predictable name (one per unit, not a random suffix) is
        # deliberate: sudoers cannot match a wildcarded argument on some
        # sudo builds/policies ("wildcards are not allowed in command
        # arguments"), so a NOPASSWD rule for this exact path only works if
        # the path is exact and stable. Concurrency isn't a real concern
        # here -- this cascade is always a single serial sequence, never
        # parallel writers to the same unit.
        temp_path = Path(tempfile.gettempdir()) / f"llmb-ollama-kv-pending-{self.unit}.conf"
        temp_path.write_bytes(content)
        try:
            self._run(["install", "-d", "-m", "0755", str(self.dropin_dir)], privileged=True)
            self._run(["install", "-m", "0644", str(temp_path), str(self.dropin_path)], privileged=True)
        finally:
            temp_path.unlink(missing_ok=True)

    def _effective_drop_in_paths(self) -> List[str]:
        result = self._run(
            [self.systemctl_command, "show", self.unit, "--property=DropInPaths", "--value"],
            check=False,
        )
        return [p for p in str(result.stdout or "").strip().split() if p]

    def _effective_environment_text(self) -> str:
        result = self._run(
            [self.systemctl_command, "show", self.unit, "--property=Environment", "--value"],
            check=False,
        )
        return str(result.stdout or "").strip()

    def _effective_kv_value(self) -> Optional[str]:
        match = _KV_ENV_RE.search(self._effective_environment_text())
        return match.group(1).lower() if match else None

    def _effective_cuda_visible_devices(self) -> Optional[str]:
        match = _CUDA_ENV_RE.search(self._effective_environment_text())
        return match.group(1).strip() if match else None

    def verify_effective_environment(self, expected_kv: str) -> None:
        """Verify systemd's merged environment before touching the live process."""
        self._run([self.systemctl_command, "daemon-reload"], privileged=True, timeout=60)
        effective = self._effective_kv_value()
        if effective == expected_kv:
            return
        conflicting: List[str] = []
        unreadable: List[str] = []
        for drop_in_path in self._effective_drop_in_paths():
            if str(self.dropin_path) == drop_in_path:
                continue
            try:
                content = self._sudo_read_bytes(Path(drop_in_path))
            except ServiceControlError:
                unreadable.append(drop_in_path)
                continue
            if _KV_ENV_RE.search(content.decode(errors="replace")):
                conflicting.append(drop_in_path)
        detail_parts = []
        if conflicting:
            detail_parts.append(f"conflicting drop-in(s): {conflicting}")
        if unreadable:
            detail_parts.append(f"unreadable drop-in(s): {unreadable}")
        if not detail_parts:
            detail_parts.append("no competing drop-in assignment was found; inspect the unit file")
        raise ServiceControlError(
            f"wrote {expected_kv} to {self.dropin_path}, but systemd's merged environment "
            f"for {self.unit} resolves OLLAMA_KV_CACHE_TYPE to {effective or 'unset'}. "
            + "; ".join(detail_parts)
            + ". Aborting before restart; no live process was touched."
        )

    def verify_gpu_binding(self) -> Optional[str]:
        """Block restart when a UUID-pinned CUDA binding no longer exists.

        Numeric CUDA indices and non-NVIDIA environments are not rewritten or
        guessed. A missing ``nvidia-smi`` yields a warning string rather than a
        false claim. A stale explicit ``GPU-...`` UUID is a hard safety error.
        """
        binding = self._effective_cuda_visible_devices()
        if not binding:
            return None
        tokens = [part.strip() for part in binding.split(",") if part.strip()]
        uuid_tokens = [token for token in tokens if token.startswith("GPU-")]
        if not uuid_tokens:
            return None
        result = self._run(
            [self.nvidia_smi_command, "--query-gpu=uuid", "--format=csv,noheader"],
            check=False,
            timeout=15,
        )
        if result.returncode != 0:
            return (
                f"could not verify CUDA_VISIBLE_DEVICES={binding!r} because nvidia-smi failed; "
                "review the service binding manually before restart"
            )
        installed = {line.strip() for line in str(result.stdout or "").splitlines() if line.strip()}
        missing = [token for token in uuid_tokens if token not in installed]
        if missing:
            raise ServiceControlError(
                f"{self.unit} has stale CUDA_VISIBLE_DEVICES UUID(s) {missing}; installed GPU UUIDs are "
                f"{sorted(installed)}. Refusing to restart a service that may lose GPU access. "
                "Correct the unit manually, then rerun the repair cascade."
            )
        return None

    def _restart_and_wait(self, *, timeout_seconds: float = 30.0) -> bool:
        """Restart and wait until the intended unit owns the Ollama socket.

        ``systemctl is-active`` can become true before ``ollama serve`` has
        bound its TCP listener.  Treating that transitional state as ready
        caused false failures on the real host.  We therefore require both:

        * the unit is active; and
        * its current MainPID owns the configured Ollama port.

        A temporarily absent listener is retried until the deadline.  A
        different unit owning the endpoint is a hard conflict and fails
        immediately rather than being hidden as a startup delay.
        """
        self._last_restart_error = None
        self._run([self.systemctl_command, "daemon-reload"], privileged=True, timeout=60)
        self._run([self.systemctl_command, "restart", self.unit], privileged=True, timeout=120)
        deadline = time.monotonic() + max(1.0, timeout_seconds)
        while time.monotonic() < deadline:
            result = self._run(
                [self.systemctl_command, "is-active", "--quiet", self.unit],
                check=False,
            )
            if result.returncode == 0:
                try:
                    self._verify_owns_live_process_once()
                    return True
                except ServiceControlError as exc:
                    message = str(exc)
                    self._last_restart_error = message
                    # Only two conditions are normal startup transitions:
                    # systemd has not published MainPID yet, or Ollama has not
                    # bound the socket yet. Any visible-but-wrong listener or
                    # unit mismatch is a hard conflict and fails immediately.
                    if not self._is_transient_ownership_error(message):
                        raise
            else:
                self._last_restart_error = f"{self.unit} is not active yet"
            self._sleep(0.5)
        return False

    def _main_pid(self) -> Optional[int]:
        result = self._run(
            [self.systemctl_command, "show", self.unit, "--property=MainPID", "--value"],
            check=False,
        )
        text = str(result.stdout or "").strip()
        return int(text) if text.isdigit() and int(text) > 0 else None

    def observed_process_kv(self) -> Optional[str]:
        pid = self._main_pid()
        if not pid:
            return None
        helper_script = self.kv_read_helper_path
        if self._kv_read_helper_exists_fn(helper_script):
            # Preferred path: a real script file, one fixed path, no PID
            # wildcard needed in the argument list at all -- a bare command
            # with no arguments specified in sudoers matches any arguments
            # to that one file, sidestepping both the "wildcards not
            # allowed" and "illegal escape sequence" rejections an inline
            # sh -c string hit on this host.
            result = self._run(
                [str(helper_script), str(pid)],
                privileged=True,
                check=False,
                timeout=10,
            )
        else:
            script = (
                'tr "\\000" "\\n" < "/proc/$1/environ" '
                '| grep -m1 "^OLLAMA_KV_CACHE_TYPE=" || true'
            )
            result = self._run(
                ["sh", "-c", script, "llmb", str(pid)],
                privileged=True,
                check=False,
                timeout=10,
            )
        line = str(result.stdout or "").strip()
        if not line.startswith("OLLAMA_KV_CACHE_TYPE="):
            return None
        value = line.split("=", 1)[1].strip().lower()
        return value or None

    def set_kv_type(self, kv_type: str, *, phase: str) -> ServicePhaseResult:
        if kv_type not in _ALLOWED_KV:
            raise ValueError(f"unsupported managed KV type: {kv_type!r}")
        # Re-check ownership immediately before mutation. This catches a
        # service handover between planning and the privileged phase.
        self.verify_owns_live_process()
        self.verify_gpu_binding()
        self.snapshot_dropin()
        self.mutation_started = True
        self._install_bytes(self.dropin_content(kv_type))
        self.verify_effective_environment(kv_type)
        try:
            active = self._restart_and_wait()
        except ServiceControlError as exc:
            note = str(exc)
            result = ServicePhaseResult(phase, self.unit, kv_type, False, False, None, note)
            event = result.to_dict()
            self.events.append(event)
            if self.event_callback:
                self.event_callback(event)
            raise
        observed = self.observed_process_kv() if active else None
        verified = bool(active and observed == kv_type)
        if not active:
            note = (
                f"{self.unit} did not become ready on TCP port {self.port} within the restart timeout"
                + (f"; last check: {self._last_restart_error}" if self._last_restart_error else "")
            )
        elif verified:
            note = f"running Ollama process verified with OLLAMA_KV_CACHE_TYPE={kv_type}"
        else:
            note = f"Ollama restart completed but live process KV value was {observed or 'unavailable'}"
        result = ServicePhaseResult(phase, self.unit, kv_type, active, verified, observed, note)
        event = result.to_dict()
        self.events.append(event)
        if self.event_callback:
            self.event_callback(event)
        if not active or not verified:
            raise ServiceControlError(note)
        return result

    def restore(self, *, phase: str = "restore") -> ServicePhaseResult:
        snapshot = self.snapshot_dropin()
        if snapshot.existed:
            assert snapshot.content is not None
            self._install_bytes(snapshot.content)
        else:
            self._run(["rm", "-f", str(self.dropin_path)], privileged=True)
        try:
            active = self._restart_and_wait()
        except ServiceControlError as exc:
            result = ServicePhaseResult(phase, self.unit, None, False, False, None, str(exc))
            event = result.to_dict()
            self.events.append(event)
            if self.event_callback:
                self.event_callback(event)
            raise
        observed = self.observed_process_kv() if active else None
        note = (
            "original llmb drop-in state restored and Ollama restarted with verified endpoint ownership"
            if active
            else (
                f"original drop-in state restored but {self.unit} did not become ready on TCP port {self.port}"
                + (f"; last check: {self._last_restart_error}" if self._last_restart_error else "")
            )
        )
        result = ServicePhaseResult(
            phase=phase,
            unit=self.unit,
            kv_type=observed,
            active=active,
            verified=active,
            observed_kv_type=observed,
            note=note,
        )
        event = result.to_dict()
        self.events.append(event)
        if self.event_callback:
            self.event_callback(event)
        if not active:
            raise ServiceControlError(result.note)
        self.mutation_started = False
        return result
