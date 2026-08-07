"""Unprivileged client for the root-owned Ollama KV repair broker.

This module deliberately contains no systemd, ``ss``, drop-in, or privileged
file-operation implementation.  The installed broker is the sole service
control boundary.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

_ALLOWED_KV = {"q8_0", "q4_0"}


class ServiceControlError(RuntimeError):
    """A broker-controlled repair operation could not be completed safely."""


@dataclass(frozen=True)
class ActiveService:
    unit: str
    pid: int
    port: int


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
        return self.__dict__.copy()


class BrokerOllamaServiceController:
    """Client facade.  Caller input never selects a unit, path, or command."""

    protocol_version = 1
    broker_path = Path("/usr/local/libexec/llmb-ollama-kv-control")

    def __init__(self, unit: str = "ollama.service", *, port: int = 11434,
                 run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
                 input_fn: Callable[[str], str] = input,
                 isatty_fn: Callable[[], bool] = lambda: bool(os.isatty(0) and os.isatty(1)),
                 sudo_command: str = "sudo", force_password_prompt: bool = True,
                 event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
                 auto_confirm: bool = False,
                 broker_path: Path = broker_path) -> None:
        if not 1 <= int(port) <= 65535:
            raise ValueError(f"invalid Ollama TCP port: {port!r}")
        self.unit, self.port = unit, int(port)  # Display-only; never sent to broker.
        self._run_impl, self._input, self._isatty = run, input_fn, isatty_fn
        self.sudo_command, self.force_password_prompt = sudo_command, bool(force_password_prompt)
        self.event_callback, self.auto_confirm, self.broker_path = event_callback, bool(auto_confirm), broker_path
        self.events: list[Dict[str, Any]] = []
        self.transaction: Optional[str] = None

    @property
    def privileged_prefix(self) -> list[str]:
        if os.geteuid() == 0:
            return []
        return [self.sudo_command, "-n"] if self.auto_confirm else [self.sudo_command]

    @classmethod
    def for_active_service(cls, **kwargs: Any) -> "BrokerOllamaServiceController":
        return cls(port=int(kwargs.pop("port", 11434)), **kwargs)

    def _call(self, *args: str) -> Dict[str, Any]:
        argv = [*self.privileged_prefix, str(self.broker_path), *args]
        try:
            result = self._run_impl(argv, capture_output=True, text=True, check=False, timeout=120)
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            raise ServiceControlError(f"privileged broker invocation failed: {exc}") from exc
        try:
            payload = json.loads(str(getattr(result, "stdout", "") or ""))
        except ValueError:
            payload = {}
        if getattr(result, "returncode", 1) != 0 or not payload.get("ok", False):
            detail = str(payload.get("message") or payload.get("error_code") or getattr(result, "stderr", "") or "broker failed")
            if self.auto_confirm and "password" in detail.lower():
                detail = "--auto-confirm requires NOPASSWD for /usr/local/libexec/llmb-ollama-kv-control: " + detail
            raise ServiceControlError(detail)
        if payload.get("protocol") != self.protocol_version:
            raise ServiceControlError("installed KV broker protocol is incompatible with this ModelBench client")
        return payload

    def require_supervised_tty(self) -> None:
        if os.geteuid() != 0 and not self._isatty():
            raise ServiceControlError("automatic Ollama service restarts require a real interactive terminal")

    def confirm(self, phase: str, message: str, *, keyword: str = "RESTART") -> None:
        if self.auto_confirm:
            print(f"\nPRIVILEGED OLLAMA SERVICE PHASE: {phase} (--auto-confirm)\n{message}")
            return
        self.require_supervised_tty()
        prompt = f"\nPRIVILEGED OLLAMA SERVICE PHASE: {phase}\n{message}\nType {keyword} to continue, or anything else to stop: "
        if self._input(prompt).strip() != keyword:
            raise ServiceControlError(f"operator declined service phase {phase!r}")

    def authorise_sudo(self) -> None:
        if os.geteuid() == 0 or self.auto_confirm:
            return
        if self.force_password_prompt:
            self._run_impl([self.sudo_command, "-k"], capture_output=False, text=True, check=False, timeout=10)
        self._run_impl([self.sudo_command, "-v"], capture_output=False, text=True, check=False, timeout=120)

    def verify_noninteractive_sudo_ready(self) -> None:
        if self.auto_confirm and os.geteuid() != 0:
            self._call("version")

    def verify_owns_live_process(self) -> ActiveService:
        if self.transaction is None:
            reply = self._call("begin", "--port", str(self.port))
            self.transaction, self.unit = str(reply["transaction"]), str(reply["unit"])
        return ActiveService(self.unit, 0, self.port)

    def verify_gpu_binding(self) -> None:
        return None

    def set_kv_type(self, kv_type: str, *, phase: str) -> ServicePhaseResult:
        if kv_type not in _ALLOWED_KV:
            raise ValueError(f"unsupported managed KV type: {kv_type!r}")
        active = self.verify_owns_live_process()
        reply = self._call("set", "--transaction", str(self.transaction), "--kv", kv_type)
        result = ServicePhaseResult(phase, active.unit, kv_type, True, bool(reply.get("verified")), reply.get("observed_kv"), "root-owned broker verified managed Ollama KV state")
        self.events.append(result.to_dict())
        if self.event_callback:
            self.event_callback(result.to_dict())
        return result

    def restore(self, *, phase: str = "restore") -> ServicePhaseResult:
        if self.transaction is None:
            return ServicePhaseResult(phase, self.unit, None, True, True, None, "no broker transaction was started")
        reply = self._call("restore", "--transaction", self.transaction)
        result = ServicePhaseResult(phase, str(reply.get("unit") or self.unit), None, True, bool(reply.get("restored")), None, "root-owned broker restored original service state")
        self.events.append(result.to_dict())
        if self.event_callback:
            self.event_callback(result.to_dict())
        self.transaction = None
        return result


# Compatibility name intentionally delegates to the only safe implementation.
OllamaServiceController = BrokerOllamaServiceController


def discover_active_service(**_kwargs: Any) -> ActiveService:
    raise ServiceControlError("Ollama service discovery is performed only by the privileged KV broker")


def discover_active_unit(**kwargs: Any) -> str:
    return discover_active_service(**kwargs).unit
