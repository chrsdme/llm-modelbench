"""Read-only local runtime discovery and reusable runtime profiles for RC21."""
from __future__ import annotations

import json
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .decision_policy import Action, DecisionPolicy
from .hardware import detect_gpus

PROFILE_SCHEMA_VERSION = 1
_PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_BACKENDS = {"ollama", "llama_cpp"}
_HEALTH = {"healthy", "unhealthy", "unreachable", "unsupported", "unknown"}
# Local health endpoints are intentionally bounded: inventories can be larger
# than a few KiB, but discovery must not read an unbounded response.
MAX_HEALTH_RESPONSE_BYTES = 4 * 1024 * 1024

# Names only: this patch intentionally does not create services or profile-store
# entries. Operators bind these to physical UUIDs and endpoints explicitly.
TOPOLOGY_PROFILE_NAMES = ("ollama-gpu0", "ollama-gpu1", "ollama-dual", "llama_cpp-single-gpu", "llama_cpp-dual-gpu")


class RuntimeProfileError(RuntimeError):
    pass


class RuntimeSelectionError(RuntimeProfileError):
    """Selection failure with a stable reason for compatibility decisions."""

    def __init__(self, message: str, *, reason: str = "selection_failed") -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    backend: str
    endpoint: str
    mode: str = "external"
    physical_gpu_uuids: Tuple[str, ...] = ()
    description: Optional[str] = None
    provenance: str = "configured"

    def __post_init__(self) -> None:
        if not _PROFILE_NAME.fullmatch(self.name):
            raise RuntimeProfileError("profile name must contain only letters, digits, '.', '_', and '-'")
        if self.backend not in _BACKENDS:
            raise RuntimeProfileError("profile backend must be 'ollama' or 'llama_cpp'")
        if self.mode != "external":
            raise RuntimeProfileError("RC21 Stage 4 supports external runtime profiles only")
        object.__setattr__(self, "endpoint", normalize_endpoint(self.endpoint))
        object.__setattr__(self, "physical_gpu_uuids", tuple(dict.fromkeys(
            str(value).strip() for value in self.physical_gpu_uuids if str(value).strip()
        )))

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["physical_gpu_uuids"] = list(self.physical_gpu_uuids)
        return value

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "RuntimeProfile":
        if not isinstance(value, dict):
            raise RuntimeProfileError("runtime profile must be an object")
        return cls(
            name=str(value.get("name") or ""), backend=str(value.get("backend") or ""),
            endpoint=str(value.get("endpoint") or ""), mode=str(value.get("mode") or "external"),
            physical_gpu_uuids=tuple(value.get("physical_gpu_uuids") or ()),
            description=(str(value["description"]) if value.get("description") is not None else None),
            provenance=str(value.get("provenance") or "configured"),
        )


@dataclass(frozen=True)
class RuntimeCandidate:
    profile: RuntimeProfile
    health: str
    source: Tuple[str, ...]
    detail: str = ""
    recommended: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"profile": self.profile.to_dict(), "health": self.health,
                "source": list(self.source), "detail": self.detail,
                "recommended": self.recommended}


def normalize_endpoint(value: str) -> str:
    parsed = urllib.parse.urlsplit(str(value or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise RuntimeProfileError("runtime endpoint must be an http(s) URL without embedded credentials")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", "")).rstrip("/")


def profile_store_path(*, environ: Optional[Dict[str, str]] = None, home: Optional[Path] = None) -> Path:
    env = os.environ if environ is None else environ
    root = env.get("XDG_CONFIG_HOME")
    if root:
        return Path(root) / "llm-modelbench" / "runtime_profiles.json"
    return (home or Path.home()) / ".config" / "llm-modelbench" / "runtime_profiles.json"


def _load_store(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"schema_version": PROFILE_SCHEMA_VERSION, "default_profile": None, "profiles": []}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeProfileError(f"invalid runtime profile store: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise RuntimeProfileError("unsupported runtime profile store schema")
    if not isinstance(value.get("profiles"), list):
        raise RuntimeProfileError("runtime profile store profiles must be a list")
    return value


def load_profiles(path: Optional[Path] = None) -> Tuple[List[RuntimeProfile], Optional[str]]:
    value = _load_store(path or profile_store_path())
    profiles = [RuntimeProfile.from_dict(item) for item in value["profiles"]]
    names = [profile.name for profile in profiles]
    if len(names) != len(set(names)):
        raise RuntimeProfileError("runtime profile names must be unique")
    default = value.get("default_profile")
    if default is not None and default not in names:
        raise RuntimeProfileError("default runtime profile does not exist")
    return profiles, default


def _atomic_write(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def save_profile(profile: RuntimeProfile, *, path: Optional[Path] = None, replace: bool = False,
                 set_default: bool = False) -> None:
    target = path or profile_store_path()
    profiles, default = load_profiles(target)
    existing = {item.name: item for item in profiles}
    if profile.name in existing and not replace:
        raise RuntimeProfileError(f"runtime profile already exists: {profile.name}")
    existing[profile.name] = profile
    _atomic_write(target, {"schema_version": PROFILE_SCHEMA_VERSION,
                           "default_profile": profile.name if set_default else default,
                           "profiles": [existing[name].to_dict() for name in sorted(existing)]})


def delete_profile(name: str, *, path: Optional[Path] = None) -> None:
    target = path or profile_store_path()
    profiles, default = load_profiles(target)
    retained = [profile for profile in profiles if profile.name != name]
    if len(retained) == len(profiles):
        raise RuntimeProfileError(f"unknown runtime profile: {name}")
    _atomic_write(target, {"schema_version": PROFILE_SCHEMA_VERSION,
                           "default_profile": None if default == name else default,
                           "profiles": [profile.to_dict() for profile in retained]})


def implicit_ollama_profile(cfg: Any) -> RuntimeProfile:
    return RuntimeProfile("legacy-ollama", "ollama", str(cfg.ollama_url), provenance="legacy-default")


def _is_local(endpoint: str) -> bool:
    hostname = urllib.parse.urlsplit(endpoint).hostname or ""
    return hostname.lower() == "localhost" or hostname in {"127.0.0.1", "::1"}


def _http_probe(endpoint: str, path: str, timeout: float = 1.5) -> Tuple[int, str]:
    request = urllib.request.Request(endpoint + path, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
            body = response.read(MAX_HEALTH_RESPONSE_BYTES + 1)
            if len(body) > MAX_HEALTH_RESPONSE_BYTES:
                return -1, (
                    "health response exceeds bounded-size limit "
                    f"({MAX_HEALTH_RESPONSE_BYTES} bytes)"
                )
            return int(response.status), body.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read(1024).decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        return 0, str(exc.reason)
    except Exception as exc:
        return 0, repr(exc)


def probe_profile(profile: RuntimeProfile, *, http_probe: Callable[[str, str], Tuple[int, str]] = _http_probe) -> Tuple[str, str]:
    if not _is_local(profile.endpoint):
        return "unsupported", "non-local endpoint discovery is disabled"
    if profile.backend == "ollama":
        version_status, version_body = http_probe(profile.endpoint, "/api/version")
        if version_status == 0:
            return "unreachable", version_body
        if version_status == -1:
            return "unhealthy", version_body
        if not 200 <= version_status < 300:
            return "unhealthy", f"/api/version HTTP {version_status}"
        tags_status, tags_body = http_probe(profile.endpoint, "/api/tags")
        if tags_status == -1:
            return "unhealthy", tags_body
        if not 200 <= tags_status < 300:
            return "unhealthy", f"/api/tags HTTP {tags_status}"
        try:
            payload = json.loads(tags_body)
            if not isinstance(payload, dict):
                return "unhealthy", "/api/tags did not return an object"
            if not isinstance(payload.get("models"), list):
                return "unhealthy", "/api/tags did not return models"
        except json.JSONDecodeError:
            return "unhealthy", "/api/tags did not return JSON"
        return "healthy", "Ollama version/tags available"
    status, body = http_probe(profile.endpoint, "/health")
    if status == 0:
        return "unreachable", body
    if 200 <= status < 300:
        return "healthy", "llama-server health endpoint available"
    return "unhealthy", f"/health HTTP {status}"


def _process_profiles(proc_root: Path = Path("/proc")) -> List[RuntimeProfile]:
    profiles: List[RuntimeProfile] = []
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return profiles
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        lower = command.lower()
        backend = "llama_cpp" if ("llama-server" in lower or "llama_server" in lower) else ("ollama" if "ollama" in lower else None)
        if backend is None:
            continue
        match = re.search(r"(?:--port(?:=|\s+)|-p\s+)(\d{1,5})\b", command)
        port = int(match.group(1)) if match else (11434 if backend == "ollama" else 8080)
        if not 1 <= port <= 65535:
            continue
        profiles.append(RuntimeProfile(
            name=f"process-{backend}-{port}", backend=backend,
            endpoint=f"http://127.0.0.1:{port}", provenance="discovered",
            description="identified from a local process command line",
        ))
    return profiles


def _recommended(profile: RuntimeProfile, health: str, gpu_count: int) -> bool:
    if health != "healthy":
        return False
    return (gpu_count > 1 and profile.backend == "llama_cpp") or (gpu_count <= 1 and profile.backend == "ollama")


def discover_runtimes(cfg: Any, *, store_path: Optional[Path] = None,
                      process_profiles: Optional[Iterable[RuntimeProfile]] = None,
                      http_probe: Callable[[str, str], Tuple[int, str]] = _http_probe,
                      gpu_devices: Optional[Iterable[Any]] = None) -> List[RuntimeCandidate]:
    saved, _ = load_profiles(store_path)
    entries: List[Tuple[RuntimeProfile, str]] = [(profile, "saved_profile") for profile in saved]
    entries.append((implicit_ollama_profile(cfg), "legacy_default"))
    entries.extend((profile, "process") for profile in (process_profiles if process_profiles is not None else _process_profiles()))
    deduped: Dict[Tuple[str, str], Tuple[RuntimeProfile, List[str]]] = {}
    for profile, source in entries:
        key = (profile.backend, normalize_endpoint(profile.endpoint))
        if key not in deduped:
            deduped[key] = (profile, [source])
        elif source not in deduped[key][1]:
            deduped[key][1].append(source)
    devices = list(detect_gpus() if gpu_devices is None else gpu_devices)
    candidates = []
    for profile, sources in deduped.values():
        health, detail = probe_profile(profile, http_probe=http_probe)
        if health not in _HEALTH:
            health, detail = "unknown", "unrecognized health probe result"
        candidates.append(RuntimeCandidate(profile, health, tuple(sources), detail,
                                           _recommended(profile, health, len(devices))))
    return sorted(candidates, key=lambda item: (item.profile.backend, item.profile.endpoint, item.profile.name))


def _decisive_winner(viable: List[RuntimeCandidate]) -> Optional[RuntimeCandidate]:
    """A winner only exists when recommendation picks out exactly one candidate.

    A lexical or positional tie-break (e.g. ``sorted(viable)[0]``) would be
    technically deterministic but not methodologically decisive -- if two
    or more healthy candidates are equally recommended (or none are), that
    is genuine ambiguity, not a winner.
    """
    recommended = [item for item in viable if item.recommended]
    return recommended[0] if len(recommended) == 1 else None


def select_runtime(candidates: Iterable[RuntimeCandidate], *, explicit_profile: Optional[str] = None,
                   default_profile: Optional[str] = None, interactive: bool = False,
                   policy: Optional[DecisionPolicy] = None,
                   input_fn: Callable[[str], str] = input, output_fn: Callable[[str], None] = print) -> RuntimeCandidate:
    values = list(candidates)
    by_name = {item.profile.name: item for item in values}
    target = explicit_profile or default_profile
    if target:
        candidate = by_name.get(target)
        if candidate is None:
            raise RuntimeSelectionError(f"unknown runtime profile: {target}", reason="unknown_profile")
        if candidate.health != "healthy":
            raise RuntimeSelectionError(
                f"runtime profile {target!r} is {candidate.health}: {candidate.detail}",
                reason="unhealthy_profile",
            )
        return candidate
    viable = [item for item in values if item.health == "healthy"]
    if len(viable) == 1:
        return viable[0]
    if not viable:
        raise RuntimeSelectionError("no healthy local runtime candidates", reason="no_healthy_candidates")
    if not interactive:
        winner = _decisive_winner(viable)
        if winner is not None and policy is not None and policy.permits(Action.BACKEND_AUTO_SELECT):
            return winner
        choices = ", ".join(item.profile.name for item in viable)
        raise RuntimeSelectionError(
            "multiple healthy runtime profiles require --runtime-profile <name>: " + choices,
            reason="runtime_selection_ambiguous",
        )
    for index, candidate in enumerate(viable, 1):
        suffix = " (recommended)" if candidate.recommended else ""
        output_fn(f"{index}. {candidate.profile.name} [{candidate.profile.backend}] {candidate.profile.endpoint}{suffix}")
    try:
        selected = int(input_fn("Select runtime number: ").strip())
        return viable[selected - 1]
    except (ValueError, IndexError):
        raise RuntimeSelectionError("runtime selection cancelled or invalid", reason="invalid_selection")
