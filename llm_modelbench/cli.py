"""Command-line interface.

Subcommands:
  doctor           preflight environment, import path, Ollama, GPU, disk
  inventory        list local models with class, size, and predicted VRAM fit
  plan             show the exact model/task/sample plan without running
  run              run the benchmark (one model at a time); --mock runs offline with a stub
  watch            live terminal dashboard for a running or completed run
  report           (re)build reports from a run directory
  pack-subjective  collate subjective outputs for human grading
  grade            blind human grading workflow for subjective outputs
  repair           scan and recover incomplete run evidence without overwriting sources
  diff             compare two runs
  export-review    zip useful run artefacts for GPT/Claude review
  selftest         verify all scoring logic offline, no Ollama needed

Every run writes to runs/<run-id>/ so --resume can pick up an interrupted run exactly.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from . import __version__
from .config import Config
from .classify import FAMILY_ORDER, classify_model, families_for, size_gb
from .filters import parse_task_ids
from .backend import (
    BackendCapability,
    InferenceClient,
    MockBackendAdapter,
    OllamaBackendAdapter,
    require_capability,
)
from .decision_policy import DecisionPolicy
from .preflight import resolve_operational_preflight
from .runtime_profiles import (
    RuntimeProfile,
    RuntimeProfileError,
    RuntimeSelectionError,
    delete_profile,
    discover_runtimes,
    implicit_ollama_profile,
    load_profiles,
    profile_store_path,
    save_profile,
    select_runtime,
)


def _ollama_port(url: str) -> int:
    parsed = urlparse(url if "://" in url else f"http://{url}")
    try:
        return int(parsed.port or 11434)
    except ValueError as exc:
        raise SystemExit(f"invalid Ollama URL port in {url!r}") from exc


def _client(args, cfg: Config, *, gpu_inventory=None) -> InferenceClient:
    """``gpu_inventory``, when supplied, is a caller's already-resolved GPU
    detection (Anvil Stage 1.3's composed preflight) reused here instead of
    triggering a second ``detect_gpus()`` call for the same invocation."""
    if getattr(args, "mock", False):
        from .ollama import MockClient
        return MockBackendAdapter(MockClient(cfg.ollama_url, cfg.seed, cfg.temperature, cfg.request_timeout))
    unattended = bool(getattr(args, "unattended", False))
    # --unattended is a decision policy, not a synonym for --yes: it only ever
    # authorizes a *decisive* backend auto-select (select_runtime's own
    # _decisive_winner() gate still fails closed on genuine ties/ambiguity),
    # and it never implies --auto-confirm/--force/privileged authority --
    # those remain their own explicit flags, untouched here.
    policy = DecisionPolicy(unattended=unattended, allow_backend_auto_selection=unattended)
    explicit = getattr(args, "runtime_profile", None)
    try:
        profiles, default = load_profiles(_runtime_store(args))
        preflight = resolve_operational_preflight(
            cfg, explicit_profile=explicit, default_profile=default,
            interactive=bool(sys.stdin.isatty()) and not unattended,
            policy=policy, store_path=_runtime_store(args), gpu_inventory=gpu_inventory,
            discover_fn=discover_runtimes, select_fn=select_runtime,
        )
        if preflight.blocked:
            blocker = preflight.blocker
            # Preserve the established no-profile Ollama behavior when the local
            # service is down. Any explicit, saved, or ambiguous selection is
            # fail-closed and must not silently choose Ollama.
            if (blocker.reason != "no_healthy_candidates" or explicit or default or profiles):
                raise RuntimeSelectionError(blocker.detail, reason=blocker.reason)
            selected = type("LegacySelection", (), {"profile": implicit_ollama_profile(cfg)})()
        else:
            selected = preflight.selected_candidate
    except RuntimeProfileError as exc:
        raise SystemExit(f"runtime selection failed: {exc}") from exc
    profile = selected.profile
    setattr(args, "_runtime_profile", profile)
    if profile.backend == "llama_cpp":
        from .llama_cpp import LlamaCppBackendAdapter, LlamaCppClient
        return LlamaCppBackendAdapter(LlamaCppClient(profile.endpoint, cfg.seed, cfg.temperature, cfg.request_timeout))
    cfg.ollama_url = profile.endpoint
    from .ollama import OllamaClient
    return OllamaBackendAdapter(OllamaClient(profile.endpoint, cfg.seed, cfg.temperature, cfg.request_timeout))


def _runtime_store(args) -> Path:
    value = getattr(args, "runtime_profiles_file", None)
    return Path(value) if value else profile_store_path()


def _resolve_and_materialise_for_run(args, cfg: Config, *, inventory=None):
    """Anvil Stage 3B.3D: the composed resolve_runtime -> materialise ->
    lifecycle-controller step for a real (non-mock) benchmark run.

    Runs the same composed discovery/selection preflight ``_client`` uses
    (one discovery, not two), converts its output into ``resolve_runtime``
    inputs, and hands the accepted resolution to the Stage 3B.3C materialiser
    (external reuse, or -- for ``llama_cpp`` with no reusable external and a
    complete recipe -- an owned ephemeral ``llama-server``). Ollama is
    reuse-only; there is no fallback between backends.

    Returns a :class:`runtime_materialisation.RuntimeMaterialisationOutcome`.
    A non-``ok`` outcome is a structured pre-row refusal the caller turns into
    a ``SystemExit`` -- never a model-quality failure.
    """
    from datetime import datetime as _dt, timezone as _tz
    from .runtime_materialisation import production_seams, resolve_and_materialise_runtime
    from .runtime_profiles import discover_backend_executables
    from .hardware import host_memory_snapshot

    unattended = bool(getattr(args, "unattended", False))
    policy = DecisionPolicy(unattended=unattended, allow_backend_auto_selection=unattended)
    explicit = getattr(args, "runtime_profile", None)
    try:
        profiles, default = load_profiles(_runtime_store(args))
        preflight = resolve_operational_preflight(
            cfg, explicit_profile=explicit, default_profile=default,
            interactive=bool(sys.stdin.isatty()) and not unattended,
            policy=policy, store_path=_runtime_store(args), gpu_inventory=inventory,
            discover_fn=discover_runtimes, select_fn=select_runtime,
        )
    except RuntimeProfileError as exc:
        raise SystemExit(f"runtime selection failed: {exc}") from exc

    # Anvil Stage 3B.4: an artifact resolved during managed-launch
    # eligibility on the blocked path, reused below to avoid a second
    # full-file content hash.
    _managed_artifact = None

    if preflight.blocked:
        blocker = preflight.blocker
        # Stage 3B.3D behaviour change (owner-visible): the pre-3B.3D no-profile
        # path built an Ollama client anyway when the local service was down and
        # let the failure surface as per-row harness errors. Under the Stage
        # 3B.3D Ollama policy (reuse-only, no spawn, no restart, no fallback) a
        # real run with no usable endpoint is a *structured pre-row refusal*
        # instead: resolve_runtime is still driven (via the implicit Ollama
        # profile, which also sets args._runtime_profile), returns
        # NO_USABLE_ENDPOINT, and the caller turns that into a clean SystemExit
        # with zero benchmark rows. `--mock` is unaffected (it never reaches
        # here). An explicit/saved/default profile keeps the earlier
        # RuntimeMaterialisationOutcome refusal shape below.
        # Anvil Stage 3B.4 managed-spawn reachability: an explicit/default
        # `llama_cpp` runtime profile whose endpoint is not serving must not
        # be a dead end. The operator declared `llama_cpp` intent; discovery
        # yields an UNHEALTHY `llama_cpp` candidate for that profile. When a
        # single explicit `--models` entry resolves to a local GGUF and the
        # `llama-server` executable is installed, the run is managed-launch
        # eligible: fall through to resolve_runtime with `selected_backend =
        # "llama_cpp"` and the unhealthy candidate list -- the resolver's
        # §2 managed-launch branch places the workload and builds an owned
        # recipe, and materialise() spawns (never reuses the dead endpoint,
        # never falls back). Any other shape keeps the structured refusal.
        _target = explicit or default
        _managed_profile = next(
            (p for p in profiles if p.name == _target and p.backend == "llama_cpp"),
            None,
        ) if _target else None
        _managed_eligible = False
        if _managed_profile is not None:
            _managed_artifact = _resolve_managed_llama_artifact(args, cfg, "llama_cpp")
            _peek_exes = None
            try:
                _peek_exes = discover_backend_executables()
            except Exception:  # noqa: BLE001 -- best-effort input
                _peek_exes = None
            _llama_installed = any(
                getattr(e, "backend", None) == "llama_cpp"
                and getattr(e, "state", None) == "installed"
                for e in (_peek_exes or ())
            )
            _managed_eligible = bool(
                _managed_artifact.ok and _llama_installed and preflight.topology
            )
        if not _managed_eligible:
            if blocker.reason != "no_healthy_candidates" or explicit or default or profiles:
                from .runtime_materialisation import RuntimeMaterialisationOutcome
                return RuntimeMaterialisationOutcome(
                    ok=False, backend="", resolution_status="no_usable_endpoint",
                    refusal_reason=f"runtime_not_resolved: no_usable_endpoint: {blocker.detail}",
                )
            selected_backend = "ollama"
            selected_endpoint_profile = implicit_ollama_profile(cfg)
            explicit_profile_name = None
        else:
            selected_backend = "llama_cpp"
            selected_endpoint_profile = _managed_profile
            explicit_profile_name = _managed_profile.name
    else:
        selected = preflight.selected_candidate
        selected_backend = selected.profile.backend
        selected_endpoint_profile = selected.profile
        explicit_profile_name = explicit or (
            selected.profile.name if selected.profile.name else None
        )

    # Read-only metadata; args._runtime_profile is the profile identity the
    # rest of cmd_run (runtime-identity collection) expects.
    setattr(args, "_runtime_profile", selected_endpoint_profile)

    backend_executables = None
    try:
        backend_executables = discover_backend_executables()
    except Exception:  # noqa: BLE001 -- executable discovery is best-effort input
        backend_executables = None

    def now_iso() -> str:
        return _dt.now(_tz.utc).isoformat()

    # Anvil Stage 3B.3E: for a managed llama_cpp spawn, resolve the selected
    # benchmark model to an authoritative *local* GGUF path + a SHA-256 hashed
    # from the actual bytes. A managed llama-server serves exactly one model,
    # and this runs before the client, so the only model reference available
    # is a single explicit `--models` entry (owner decision 3B.3E-OD1). Any
    # other shape -- multi-model, all-installed default, --select, a non-llama
    # backend -- leaves the artifact unresolved.
    # Reuse the artifact already resolved during managed-launch eligibility
    # (same inputs -> same result; avoids a second full-file content hash of
    # a multi-GB GGUF on the managed-launch-from-blocked path).
    if _managed_artifact is not None and selected_backend == "llama_cpp":
        artifact = _managed_artifact
    else:
        artifact = _resolve_managed_llama_artifact(args, cfg, selected_backend)
    model_path = artifact.resolved_path if artifact.ok else None
    model_sha = artifact.verified_sha256 if artifact.ok else None

    # Anvil Stage 3B.4: ModelBench makes a managed GPU-placement decision only
    # for `llama_cpp` with a resolved local GGUF -- the one path where an
    # owned ephemeral llama-server can be spawned from the resolved recipe.
    # Ollama (amendment §23 reuse-only) and `llama_cpp` with no resolved GGUF
    # are reuse-only: the resolver skips fit/placement (there is nothing to
    # place), returns a reuse-only RESOLVED resolution, and materialisation
    # may only reuse an already-running external endpoint (never spawn). The
    # workload estimate for the managed path is the verified GGUF file size --
    # the conservative benchmark-required weight estimate for materialising
    # exactly those bytes (owner decision 3B.3E-OD3 / WORKLOAD ESTIMATE
    # POLICY); it comes from the same path whose content SHA was verified.
    owned_placement = selected_backend == "llama_cpp" and artifact.ok
    weight_bytes = artifact.size_bytes if owned_placement else None
    weight_bytes_source = (
        "verified_local_gguf_file_size" if owned_placement else None
    )
    requested_context = getattr(cfg, "ctx_override", None)
    kv_cache_bytes = None
    kv_cache_bytes_source = "not_required_reuse_only"
    if owned_placement:
        kv_cache_bytes, kv_cache_bytes_source = _managed_kv_cache_estimate(
            artifact.resolved_path, requested_context,
        )
    artifact_snapshot = _artifact_resolution_snapshot(
        artifact, selected_backend, owned_placement=owned_placement,
        weight_bytes=weight_bytes, weight_bytes_source=weight_bytes_source,
        requested_context=requested_context,
        kv_cache_bytes=kv_cache_bytes,
        kv_cache_bytes_source=kv_cache_bytes_source,
    )

    seams = production_seams(
        executable_path=_llama_server_executable_path(backend_executables),
        model_path=model_path,
        model_primary_sha256=model_sha,
        hardware_inventory=list(preflight.gpu_inventory),
        now_iso=now_iso,
    )
    outcome = resolve_and_materialise_runtime(
        selected_backend=selected_backend,
        discovered_candidates=list(preflight.candidates),
        topology=preflight.topology,
        host_meminfo=host_memory_snapshot() or {},
        seams=seams,
        weight_bytes=weight_bytes,
        allow_ram_spill=bool(getattr(cfg, "allow_ram_spill", False)),
        requested_context=requested_context,
        kv_cache_bytes=kv_cache_bytes,
        explicit_profile_name=explicit_profile_name,
        backend_executables=backend_executables,
        model_primary_sha256=model_sha,
        owned_placement_required=owned_placement,
        artifact_resolution=artifact_snapshot,
    )
    return outcome


def _resolve_managed_llama_artifact(args, cfg: Config, selected_backend):
    """Stage 3B.3E local GGUF artifact resolution for the managed llama_cpp
    path. Returns a :class:`local_artifact_resolver.LocalArtifactResolution`
    -- always, never raising: a not-``ok`` result (including NO_MODEL_REF for a
    non-llama backend / multi-model / default selection) leaves the managed
    spawn fail-closed while external / Ollama reuse proceeds untouched."""
    from .local_artifact_resolver import resolve_local_gguf_artifact
    from .selection import parse_models_spec

    if selected_backend != "llama_cpp":
        return resolve_local_gguf_artifact(None)
    try:
        requested = parse_models_spec(getattr(args, "models", None))
    except ValueError:
        requested = None
    single = requested[0] if requested and len(requested) == 1 else None
    return resolve_local_gguf_artifact(
        single,
        artifacts_map=getattr(cfg, "gguf_artifacts", None),
        root_dir=getattr(cfg, "gguf_root", None),
    )


def _managed_kv_cache_estimate(resolved_path: "str | None", requested_context: "int | None") -> "tuple[int | None, str]":
    """Anvil Stage 3B.4 corrective (third): a pre-spawn KV-cache byte
    estimate for a managed llama_cpp launch, derived from the resolved
    local GGUF file's own header -- never from a live backend, never
    invented when no concrete context is set.

    Returns ``(None, reason)`` for every case the managed resolver must
    then refuse closed at :attr:`RuntimeResolutionStatus.FIT_UNKNOWN`
    (Anvil Stage 3B.4 corrective #2): no resolved path, no concrete
    ``requested_context`` (this function never substitutes a default), an
    unreadable/malformed/unsupported-architecture GGUF header, or an
    architecture whose declared inputs :func:`calculate_kv_cache_bytes`
    itself rejects (e.g. the requested context exceeding the model's own
    maximum). ``reason`` is always a short machine-stable string, recorded
    verbatim as ``kv_cache_bytes_source`` evidence regardless of outcome.
    """
    if resolved_path is None:
        return None, "no_resolved_gguf_artifact"
    if requested_context is None:
        return None, "no_concrete_requested_context"
    from .gguf_metadata import resolve_gguf_architecture
    from .runtime_fit import RuntimeFitModel, calculate_kv_cache_bytes

    gguf_arch = resolve_gguf_architecture(resolved_path)
    if not gguf_arch.ok:
        return None, f"gguf_metadata_{gguf_arch.status.value}"
    model = RuntimeFitModel(
        name=gguf_arch.architecture_name or "unknown",
        weight_bytes=None,
        weight_provenance="not_evaluated_here",
        requested_context=int(requested_context),
        model_max_context=gguf_arch.model_max_context,
        architecture=gguf_arch.architecture,
    )
    kv_cache_bytes, kv_reason = calculate_kv_cache_bytes(model)
    if kv_cache_bytes is None:
        return None, f"gguf_metadata_resolved_but_{kv_reason}"
    return kv_cache_bytes, kv_reason


def _artifact_resolution_snapshot(
    artifact, selected_backend, *, owned_placement, weight_bytes,
    weight_bytes_source, requested_context, kv_cache_bytes=None,
    kv_cache_bytes_source="unknown_component_not_a_gate",
) -> dict:
    """A JSON-serialisable evidence snapshot of the Stage 3B.3E artifact
    resolution + the Stage 3B.4 workload-estimate decision.

    ``owned_placement`` -- whether ModelBench makes a managed GPU-placement
    decision for this run (``llama_cpp`` + resolved local GGUF). When
    ``False`` the run is reuse-only: no fit gate, no spawn, no workload
    estimate required."""
    snapshot = artifact.to_dict()
    snapshot["owned_placement"] = bool(owned_placement)
    snapshot["reuse_only"] = not bool(owned_placement)
    if owned_placement:
        snapshot["workload_estimate_status"] = "supplied"
        snapshot["weight_bytes"] = weight_bytes
        snapshot["weight_bytes_source"] = weight_bytes_source
        snapshot["kv_cache_bytes"] = kv_cache_bytes
        snapshot["kv_cache_bytes_source"] = kv_cache_bytes_source
        snapshot["requested_context"] = requested_context
        snapshot["requested_context_source"] = (
            "operator_ctx_override" if requested_context is not None
            else "model_metadata_default_fit_off"
        )
    else:
        snapshot["workload_estimate_status"] = "not_required_reuse_only"
        snapshot["compatibility_basis"] = (
            "candidate_health + external_still_healthy"
        )
    # Stage 3B.4: a reuse-only run (Ollama, or llama_cpp with no resolved
    # GGUF) is no longer a "blocked managed spawn" -- it reaches
    # REUSED_EXTERNAL or a structured reuse refusal, never
    # RESOLVED_RECIPE_INCOMPLETE for a missing artifact.
    snapshot["blocked_managed_spawn"] = False
    return snapshot


def _llama_server_executable_path(backend_executables) -> "str | None":
    if not backend_executables:
        return None
    for entry in backend_executables:
        if getattr(entry, "backend", None) == "llama_cpp":
            return getattr(entry, "executable_path", None)
    return None


def _client_for_materialised_endpoint(endpoint: str, cfg: Config, *, backend: str) -> InferenceClient:
    """Build the backend client against the materialisation-verified endpoint
    (never a stale ``cfg.ollama_url`` / configured profile endpoint). The
    backend is the resolver's -- it is not re-decided here."""
    if backend == "llama_cpp":
        from .llama_cpp import LlamaCppBackendAdapter, LlamaCppClient
        return LlamaCppBackendAdapter(
            LlamaCppClient(endpoint, cfg.seed, cfg.temperature, cfg.request_timeout)
        )
    cfg.ollama_url = endpoint
    from .ollama import OllamaClient
    return OllamaBackendAdapter(
        OllamaClient(endpoint, cfg.seed, cfg.temperature, cfg.request_timeout)
    )


def _observed_rss_bytes(materialisation) -> "int | None":
    """Anvil Stage 3B.5 -- a bounded, one-shot ``/proc/<pid>/status`` read of
    the owned managed runtime's resident set, for evidence only. ``None`` for
    external/Ollama reuse (no owned pid) or if the process cannot be read
    (already gone, permission denied) -- never raises."""
    controller = getattr(materialisation, "controller", None)
    if controller is None or not getattr(controller, "owns_runtime", False):
        return None
    outcome = getattr(materialisation, "materialisation", None)
    result = getattr(outcome, "result", None) if outcome is not None else None
    owned = getattr(result, "owned_runtime", None) if result is not None else None
    launch_proof = getattr(owned, "launch_proof", None) if owned is not None else None
    pid = getattr(launch_proof, "pid", None) if launch_proof is not None else None
    if pid is None:
        return None
    from . import runtime_process_linux as rpl

    return rpl.read_process_rss_bytes(pid)


def _runtime_telemetry_ref(out_dir: Path) -> "dict | None":
    """Anvil Stage 3B.5 -- a plain file-existence + field read of the
    separate, pre-existing ``runtime_telemetry.json`` artifact that
    ``runner.run(capture_runtime_telemetry=True)`` already writes (see
    DEFECT-3B.5-01). Never re-collects telemetry; a read failure or missing
    file returns ``None`` (the artifact genuinely was not written, e.g. a
    pre-benchmark refusal never reaches ``runner.run()`` at all)."""
    path = out_dir / "runtime_telemetry.json"
    try:
        if not path.exists():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict):
        return None
    return {"artifact": "runtime_telemetry.json", "status": value.get("status")}


def _persist_materialisation_evidence(
    out_dir: Path, materialisation, *, benchmark_completed: bool, failure_stage: str | None = None,
    rss_bytes_launch_ready: "int | None" = None, rss_bytes_post_execution: "int | None" = None,
) -> None:
    """Write ``materialisation_evidence.json`` after the lifecycle scope has
    exited. Observes the (structurally-recorded, never-raised) cleanup outcome
    so a clean benchmark with a failed cleanup is not lost -- persisted as an
    evidence field and surfaced as a printed warning.

    ``failure_stage`` names the phase that raised *after* an owned runtime had
    been materialised but *before* the benchmark ran -- ``"client_construction"``
    (building the backend client) or ``"pre_run_gates"`` (runtime-identity /
    resume-identity / ranking-scope / plan / confirmation). It is persisted so
    such a failure is never later mistaken for a model-quality failure. Only
    ``"client_construction"`` additionally arms a cleanup-failure warning today
    (the ``"pre_run_gates"`` cleanup-failure-warning gap is recorded as accepted
    debt -- see ANVIL_PROGRESS DEFECT-3B.3D-03).

    ``rss_bytes_launch_ready`` / ``rss_bytes_post_execution`` are the caller's
    own inline ``/proc`` samples (Anvil Stage 3B.5) -- this function does not
    read ``/proc`` itself; by the time it runs (the outer ``finally``, after
    the lifecycle ``with`` block has already torn an owned process down) the
    pid may no longer exist."""
    from .runtime_materialisation import materialisation_evidence

    controller = getattr(materialisation, "controller", None)
    cleanup_result = getattr(controller, "last_cleanup", None) if controller is not None else None
    record = materialisation_evidence(
        materialisation, cleanup_result=cleanup_result,
        benchmark_completed=benchmark_completed, failure_stage=failure_stage,
        rss_bytes_launch_ready=rss_bytes_launch_ready,
        rss_bytes_post_execution=rss_bytes_post_execution,
        runtime_telemetry_ref=_runtime_telemetry_ref(out_dir),
    )
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "materialisation_evidence.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        print(f"warning: could not persist materialisation evidence: {exc}")
    warnings = record.get("warnings", [])
    detail = (record.get("cleanup") or {}).get("detail") or "see materialisation_evidence.json"
    if "cleanup_failed_on_successful_benchmark" in warnings:
        print(
            "warning: the benchmark completed but the owned runtime cleanup "
            f"did not succeed ({detail}); the managed llama-server may still be "
            "running -- check and stop it manually."
        )
    if "cleanup_failed_after_client_construction_failure" in warnings:
        print(
            "warning: client construction failed before the benchmark ran and "
            f"the owned runtime cleanup then also did not succeed ({detail}); "
            "the managed llama-server may still be running -- check and stop it "
            "manually."
        )
    if "cleanup_failed_before_benchmark" in warnings:
        print(
            "warning: the run was refused before the benchmark started and the "
            f"owned runtime cleanup then also did not succeed ({detail}); the "
            "managed llama-server may still be running -- check and stop it "
            "manually."
        )


def _confirm_profile_change(message: str, *, yes: bool) -> None:
    if yes:
        return
    if not sys.stdin.isatty():
        raise SystemExit(message + " Re-run with --yes after reviewing the change.")
    if input(message + " Type y to continue: ").strip().lower() not in {"y", "yes"}:
        raise SystemExit("runtime profile change cancelled")


def cmd_runtime(args, cfg) -> None:
    """Keep runtime-management failures at the CLI boundary, not as tracebacks."""
    try:
        _cmd_runtime(args, cfg)
    except RuntimeProfileError as exc:
        raise SystemExit(str(exc)) from exc


def _cmd_runtime(args, cfg) -> None:
    store = _runtime_store(args)
    if args.runtime_cmd == "list":
        profiles, default = load_profiles(store)
        print(json.dumps({"store": str(store), "default_profile": default,
                          "profiles": [profile.to_dict() for profile in profiles]}, indent=2))
        return
    if args.runtime_cmd == "show":
        profiles, default = load_profiles(store)
        profile = next((item for item in profiles if item.name == args.name), None)
        if profile is None:
            raise SystemExit(f"unknown runtime profile: {args.name}")
        print(json.dumps({"profile": profile.to_dict(), "default": default == profile.name}, indent=2))
        return
    if args.runtime_cmd == "discover":
        print(json.dumps([candidate.to_dict() for candidate in discover_runtimes(cfg, store_path=store)], indent=2))
        return
    if args.runtime_cmd == "save":
        profile = RuntimeProfile(
            args.name, args.backend, args.endpoint, physical_gpu_uuids=tuple(args.gpu_uuid or ()),
            description=args.description, provenance=args.provenance,
        )
        profiles, _ = load_profiles(store)
        existing = any(item.name == profile.name for item in profiles)
        if existing and not args.replace:
            raise SystemExit(f"runtime profile already exists: {profile.name}; use --replace to replace it")
        if existing:
            _confirm_profile_change(f"Replace saved runtime profile {profile.name!r}?", yes=args.yes)
        save_profile(profile, path=store, replace=args.replace, set_default=args.set_default)
        print(json.dumps(profile.to_dict(), indent=2))
        return
    if args.runtime_cmd == "delete":
        profiles, _ = load_profiles(store)
        if not any(profile.name == args.name for profile in profiles):
            raise RuntimeProfileError(f"unknown runtime profile: {args.name}")
        _confirm_profile_change(f"Delete saved runtime profile {args.name!r}? This does not affect a runtime service or models.", yes=args.yes)
        delete_profile(args.name, path=store)
        print(f"deleted runtime profile {args.name}")
        return
    if args.runtime_cmd == "select":
        profiles, default = load_profiles(store)
        candidates = discover_runtimes(cfg, store_path=store)
        try:
            selected = select_runtime(
                candidates, explicit_profile=args.runtime_profile, default_profile=default,
                interactive=bool(sys.stdin.isatty()),
            )
        except RuntimeSelectionError as exc:
            raise SystemExit(str(exc)) from exc
        profile = selected.profile
        if args.save_name:
            saved = RuntimeProfile(
                args.save_name, profile.backend, profile.endpoint,
                physical_gpu_uuids=profile.physical_gpu_uuids,
                description=profile.description, provenance="discovered",
            )
            existing = any(item.name == saved.name for item in profiles)
            if existing and not args.replace:
                raise SystemExit(f"runtime profile already exists: {saved.name}; use --replace to replace it")
            if existing:
                _confirm_profile_change(f"Replace saved runtime profile {saved.name!r}?", yes=args.yes)
            save_profile(saved, path=store, replace=args.replace, set_default=args.set_default)
            profile = saved
        print(json.dumps({"selected": profile.to_dict(), "recommended": selected.recommended}, indent=2))
        return


def _run_dir(args) -> Path:
    rid = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(args.out or "runs") / rid


def _require_run_dir(args, *, command: str) -> Path:
    if getattr(args, "run_id", None):
        base = getattr(args, "runs_dir", None) or getattr(args, "out", None) or "runs"
        return Path(base) / str(args.run_id)
    if getattr(args, "out", None):
        return Path(args.out)
    raise SystemExit(f"{command} requires --run-id or --out")


def _resolve_model_selection(args, client):
    from .selection import parse_models_spec, resolve_exact_models, select_models
    installed = [row.get("name") for row in client.tags() if row.get("name")]
    requested = parse_models_spec(getattr(args, "models", None))
    if requested is not None:
        return resolve_exact_models(requested, installed)
    if getattr(args, "select", False):
        return select_models(installed)
    # --all is explicit documentation of the default all-installed behavior.
    # There is deliberately no -all alias.
    return None


def _apply_managed_owned_model_selection(args, client, materialisation) -> None:
    """For an owned managed llama_cpp run backed by a resolved local GGUF
    artifact (Stage 3B.3E ``gguf_artifacts``/``gguf_root``), a managed
    llama-server serves exactly one model and reports its own name/id
    through ``/v1/models`` -- never the original ``--models`` ref (an HF/
    Ollama-shaped string) used to pick the artifact file.
    ``_resolve_model_selection`` -> ``resolve_exact_models`` requires an
    exact/case-insensitive/fuzzy match against ``client.tags()``, so the
    original ref is unconditionally rejected as "not an installed model"
    even though materialisation just proved it resolves to exactly one
    local artifact and that artifact is now being served.

    This sets ``args._selected_models`` (the same pre-resolution seam used
    by interactive/campaign selection, read at the top of ``cmd_run``) to
    the single served name in that one case, so exact-selection semantics
    are preserved end to end: one configured local artifact, one served
    model, one selection -- resolved by identity (the artifact this exact
    run just materialised), not by scanning or fuzzy-matching. Does not
    touch Ollama or external llama-server reuse: it only fires when
    ``artifact_resolution.owned_placement`` is true, which is exclusively
    the managed local-GGUF spawn path."""
    if getattr(args, "_selected_models", None) is not None:
        return
    snapshot = getattr(materialisation, "artifact_resolution", None) or {}
    if not snapshot.get("owned_placement") or snapshot.get("status") != "resolved":
        return
    from .selection import parse_models_spec

    try:
        requested = parse_models_spec(getattr(args, "models", None))
    except ValueError:
        return
    if not requested or len(requested) != 1 or requested[0] != snapshot.get("model_ref"):
        return
    served = [row.get("name") for row in client.tags() if row.get("name")]
    if len(served) == 1:
        args._selected_models = served


def _confirm_destructive_compute(message: str, *, yes: bool) -> None:
    """Confirm operations that can consume substantial model time.

    This does not describe filesystem deletion; it protects unattended scripts
    from accidentally launching a long benchmark/judge batch.
    """
    if yes:
        return
    if not sys.stdin.isatty():
        raise SystemExit(message + " Non-interactive execution requires --yes after reviewing the printed plan.")
    answer = input(message + " Type y to continue: ").strip().lower()
    if answer not in {"y", "yes"}:
        raise SystemExit("cancelled before model work")


def cmd_inventory(args, cfg):
    client = _client(args, cfg)
    rows = client.tags()
    if not rows:
        print("No models found (is Ollama running?). Try: llm-modelbench inventory --mock")
        return
    from .capabilities import interrogate_model
    from .hardware import detect_gpus
    from .placement import model_placement_fit, topology_for_config
    # Anvil Stage 1.3: detect once and reuse for every model below, instead
    # of each model's fit check independently re-detecting. This is a pure
    # detection-reuse fix -- inventory remains read-only and does not go
    # through the composed operational preflight (which would additionally
    # perform backend *selection*, out of place for an inspection command).
    inventory = detect_gpus()
    topology = topology_for_config(cfg, inventory=inventory)
    items = []
    for model_row in rows:
        name = model_row.get("name", "")
        profile = interrogate_model(client, name, functional=bool(getattr(args, "auto", False)))
        families = profile.get("supported_families") or []
        fit = model_placement_fit(model_row, cfg, inventory=inventory)
        items.append({"name": name,
                      "class": classify_model(name, profile.get("declared_capabilities"), families),
                      "size_gb": size_gb(model_row), "families": families,
                      "declared_capabilities": profile.get("declared_capabilities") or [],
                      "capability_warnings": profile.get("warnings") or [],
                      "fit_classification": fit.classification,
                      "selected_gpu_uuids": list(fit.selected_gpu_uuids),
                      "will_offload": fit.classification == "confirmed_no_fit"})
    items.sort(key=lambda x: (x["class"], -x["size_gb"]))
    if args.json:
        print(json.dumps(items, indent=2)); return
    for device in topology.devices:
        now = device.effective_now_bytes
        print(f"GPU {device.uuid} safe/effective: {round(now / 1024**3, 3) if now is not None else 'unknown'} GiB")
    print(f"max single-GPU: {round(topology.max_single_effective_bytes / 1024**3, 3) if topology.max_single_effective_bytes is not None else 'unknown'} GiB")
    print(f"aggregate: {round(topology.aggregate_effective_bytes / 1024**3, 3) if topology.aggregate_effective_bytes is not None else 'unknown'} GiB")
    if topology.aggregate_policy_cap_bytes is not None:
        print(f"policy ceiling: {round(topology.aggregate_policy_cap_bytes / 1024**3, 3)} GiB")
    print()
    for it in items:
        print(f"{it['class']:<11} {it['size_gb']:>6}GB  {it['fit_classification']:<28} {it['name']}")


def cmd_capability_evidence(args, cfg):
    """Anvil Stage 2.7A: read-only classification of the fleet's stored
    capability evidence -- never reprobes, never mutates any
    ``capability_report.json``."""
    from pathlib import Path as _Path
    from .capability_evidence_classification import classify_fleet
    client = _client(args, cfg)
    report = classify_fleet(client, runs_dir=_Path(args.runs_dir), campaigns_root=_Path(args.campaigns_dir))
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
        return
    print(f"models considered: {len(report.models_considered)}")
    print(f"source files scanned: {len(report.source_files_scanned)}")
    print()
    print("by status:")
    for status, count in report.by_status().items():
        if count:
            print(f"  {status:<36} {count}")
    print()
    cells = report.reprobe_required_cells() if args.reprobe_required_only else report.cells
    label = "reprobe-required cells" if args.reprobe_required_only else "cells"
    print(f"{label}:")
    for cell in cells:
        print(f"  {cell.model:<40} {cell.capability:<10} {cell.status.value:<36} {cell.reason}")


def cmd_reprobe_plan(args, cfg):
    """Anvil Stage 2.7B: deterministic, read-only reprobe plan built from
    Stage 2.7A's evidence classification -- never runs a probe, never
    appends evidence, never mutates any stored file."""
    from pathlib import Path as _Path
    from .capability_reprobe_plan import plan_fleet_reprobes
    client = _client(args, cfg)
    plan = plan_fleet_reprobes(client, runs_dir=_Path(args.runs_dir), campaigns_root=_Path(args.campaigns_dir))
    actions = plan.filtered(
        model=args.model, capability=args.capability, backend=args.backend,
        reason=args.reason, only_required=args.only_required,
    )
    if args.json:
        payload = plan.to_dict()
        payload["actions"] = [action.to_dict() for action in actions]
        print(json.dumps(payload, indent=2))
        return
    summary = plan.summary()
    print(f"plan hash: {plan.canonical_plan_hash()}")
    print(f"models considered: {len(plan.models_considered)}")
    print(f"total cells examined: {summary['total_cells_examined']}")
    print(f"total reprobe actions: {summary['total_actions']}")
    print()
    print("by classification:")
    for classification, count in summary["by_classification"].items():
        if count:
            print(f"  {classification:<36} {count}")
    print()
    label = "reprobe actions" if args.only_required else "actions"
    print(f"{label} (after filters):")
    for action in actions:
        print(f"  {action.model:<40} {action.capability:<10} {action.action.value:<10} {action.classification:<36} {action.classification_reason}")


def cmd_reprobe_execute(args, cfg):
    """Anvil Stage 2.7C: execute only the REPROBE actions from a Stage 2.7B
    plan -- runs real functional probes, appends CapabilityObservation
    records to an EvidenceLedger (explicit SUPERSEDES where a prior native
    observation exists), and reports before/after fleet evidence metrics.
    Dry-run by default; requires --apply for any probe or ledger write."""
    from pathlib import Path as _Path
    from .capability_reprobe_execute import default_ledger_path, run_reprobe_execution
    from .capability_reprobe_plan import plan_fleet_reprobes
    from .evidence import EvidenceLedger

    client = _client(args, cfg)
    runs_dir = _Path(args.runs_dir)
    campaigns_dir = _Path(args.campaigns_dir)
    plan = plan_fleet_reprobes(client, runs_dir=runs_dir, campaigns_root=campaigns_dir)
    actions = plan.filtered(
        model=args.model, capability=args.capability, backend=args.backend,
        reason=args.reason, only_required=True,
    )
    print(f"plan hash: {plan.canonical_plan_hash()}")
    print(f"reprobe actions selected (after filters): {len(actions)}")
    for action in actions:
        print(f"  {action.model:<40} {action.capability:<10} {action.classification:<36} {action.classification_reason}")
    if not args.apply:
        print("dry-run only; no probes were run and no evidence was appended (pass --apply to execute)")
        return
    if not actions:
        print("nothing to execute: no REPROBE action survives the current filters")
        return

    ledger_path = _Path(args.ledger_path) if args.ledger_path else default_ledger_path(runs_dir)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = EvidenceLedger(ledger_path)
    report = run_reprobe_execution(
        plan, client, ledger, runs_dir=runs_dir, campaigns_root=campaigns_dir, actions=actions,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
        return
    print()
    print(f"ledger: {ledger_path}")
    native = report.native_evidence_after
    print(f"attempted: {native['attempted']}  appended: {native['appended']}  skipped: {native['skipped']}  errored: {native['errored']}")
    print(f"superseded prior records: {native['superseded_record_count']}")
    print()
    print("outcomes:")
    for outcome in report.outcomes:
        status = "OK" if outcome.appended else ("ERROR" if outcome.error else "SKIP")
        detail = outcome.after_status if outcome.appended else (outcome.error or outcome.skip_reason)
        print(f"  {status:<6} {outcome.model:<40} {outcome.capability:<10} {detail}")


def cmd_runtime_fit(args, cfg):
    """Read-only Stage 7 assessment; it does not issue a generation request."""
    from .runtime_fit import RuntimeFitProfile, collect_runtime_fit
    client = _client(args, cfg)
    selected = getattr(args, "_runtime_profile", None)
    if selected is None and getattr(args, "mock", False):
        selected = implicit_ollama_profile(cfg)
    if selected is None:
        raise SystemExit("runtime-fit requires a selected runtime profile")
    try:
        weights = {key.strip(): float(value) for item in (args.allocation_weights or "").split(",") if item.strip()
                   for key, value in (item.split("=", 1),)}
    except ValueError as exc:
        raise SystemExit("--allocation-weights must be comma-separated GPU-UUID=positive-number pairs") from exc
    try:
        profile = RuntimeFitProfile(
            selected.name, selected.backend, tuple(selected.physical_gpu_uuids),
            strategy=args.strategy, allocation_weights=weights,
            allow_cpu_spill=True if args.allow_cpu_spill else None,
        )
        result = collect_runtime_fit(client=client, model_name=args.model, profile=profile,
                                     requested_context=args.context, reserve_mib=args.reserve_mib)
    except ValueError as exc:
        raise SystemExit(f"runtime-fit refused: {exc}") from exc
    value = result.to_dict()
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
    if args.out:
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    if args.json:
        print(payload)
        return
    print(f"Runtime fit: {result.decision} ({', '.join(result.reasons)})")
    print(f"Model: {result.model.name} size={result.model.weight_bytes} bytes ({result.model.weight_provenance})")
    print(f"Profile: {result.profile.name} backend={result.profile.backend} strategy={result.profile.strategy or 'none'}")
    print(f"Reserve: {result.reserve_bytes} bytes; KV: {result.kv_cache_bytes if result.kv_cache_bytes is not None else 'unknown'} ({result.kv_provenance})")
    for item in result.device_assessments:
        print(f"  {item.gpu_uuid}: {item.decision}; installed={item.installed_capacity_bytes}; live_free={item.live_free_capacity_bytes}; {item.detail}")




def _safe_run_id(value: str) -> str:
    text = str(value or "run").strip() or "run"
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)[:160] or "run"


def _ranking_dir_for(args, *, run_id: str | None = None, fallback: str = "rankings") -> Path | None:
    if getattr(args, "no_ranking_update", False):
        return None
    explicit = getattr(args, "rankings_out", None)
    if explicit:
        return Path(explicit)
    if getattr(args, "separate_ranking", False):
        return Path("rankings-separate") / _safe_run_id(run_id or getattr(args, "run_id", None) or "diagnostic")
    return Path(fallback)


def _write_run_ranking_scope(run_dir: Path, args, *, rankings_dir: Path | None = None) -> None:
    from .ranking_controls import SCOPE_CANONICAL, SCOPE_SEPARATE, write_run_scope
    if getattr(args, "separate_ranking", False):
        write_run_scope(run_dir, scope=SCOPE_SEPARATE, rankings_dir=rankings_dir)
    else:
        write_run_scope(run_dir, scope=SCOPE_CANONICAL, rankings_dir=rankings_dir)

def _categories(args):
    return args.categories.split(",") if getattr(args, "categories", None) else None


def _task_ids(args):
    return parse_task_ids(getattr(args, "tasks", None))


_HOST_CODE_SCORERS = {"python", "filesort", "js_debounce", "fim"}


def _host_code_tasks(plan):
    """Return executable task IDs present in a rendered run plan."""
    from .tasks import TASKS

    task_by_id = {task.id: task for task in TASKS}
    task_ids = {
        str(task_id)
        for model in (plan.get("active_models") or [])
        for task_id in (model.get("tasks") or [])
    }
    return sorted(
        task_id for task_id in task_ids
        if task_id in task_by_id and task_by_id[task_id].scorer in _HOST_CODE_SCORERS
    )


def _require_host_code_opt_in(args, plan) -> None:
    tasks = _host_code_tasks(plan)
    if tasks and not bool(getattr(args, "allow_host_code_execution", False)):
        joined = ", ".join(tasks)
        raise SystemExit(
            "run refused before model execution: selected tasks execute model-generated code "
            f"on the host ({joined}). Re-run inside a disposable container/VM and add "
            "--allow-host-code-execution after reviewing docs/SAFETY.md."
        )


def _plan_for_args(args, cfg, client, *, selected_models=None, capability_profiles=None, runtime_identities=None):
    from . import planner
    return planner.build_plan(
        client, cfg,
        level=getattr(args, "level", "smoke"),
        include=getattr(args, "include_regex", None),
        exclude=getattr(args, "exclude_regex", None),
        skip_offload=getattr(args, "skip_offload", False),
        categories=_categories(args),
        task_ids=_task_ids(args),
        task_regex=getattr(args, "task_regex", None),
        family_base_only=getattr(args, "family_base_only", False),
        context_aliases_only=getattr(args, "context_aliases_only", False),
        context_only=getattr(args, "context_only", False),
        sample_mode=getattr(args, "sample_mode", "smart"),
        judge_mode=getattr(args, "judge", "off"),
        selected_models=selected_models,
        auto_probe=bool(getattr(args, "auto", False)),
        capability_profiles=capability_profiles,
        runs_dir=Path(getattr(args, "runs_dir", None) or getattr(args, "out", None) or "runs"),
        runtime_identities=runtime_identities,
    )


def _confirm_plan(args, plan):
    from . import planner
    print(planner.render_plan(plan))
    if getattr(args, "plan_json", None):
        planner.write_plan(Path(args.plan_json), plan)
        print(f"plan json -> {args.plan_json}")
    if getattr(args, "yes", False):
        return
    if not sys.stdin.isatty():
        raise SystemExit("run plan was printed but not approved. Non-interactive execution requires --yes; no benchmark task calls were made. If --auto was explicitly requested, its small capability probes may already have run while building the plan.")
    ans = input("\nProceed with this run? Type y to continue: ").strip().lower()
    if ans not in {"y", "yes"}:
        raise SystemExit("cancelled before run")

def cmd_run(args, cfg):
    from . import runner, report
    from .runtime_identity import collect_runtime_identity, validate_resume_runtime_identities
    from .hardware import detect_gpus
    if args.family_base_only and args.context_aliases_only:
        raise SystemExit("--family-base-only and --context-aliases-only cannot be used together")
    if args.samples is not None:
        cfg.samples = args.samples
    if getattr(args, "ctx", None):
        cfg.ctx_override = int(args.ctx)
    if getattr(args, "num_predict", None):
        cfg.num_predict_override = int(args.num_predict)
    if getattr(args, "think", None):
        cfg.think = args.think
    if getattr(args, "needle_max_ctx", None):
        cfg.needle_max_ctx = int(args.needle_max_ctx)
    if hasattr(args, "dump_raw"):
        cfg.dump_raw = bool(args.dump_raw)
    if hasattr(args, "fingerprint"):
        cfg.fingerprint = bool(args.fingerprint)
    if getattr(args, "judge_model", None):
        cfg.judge_model = args.judge_model
    # Anvil Stage 3.2C-2b: RAM spill is an execution-time operator permission,
    # never a persisted config field (a config file must not silently enable
    # spill -- amendment §6).  Set it only from the explicit CLI flag; a
    # config-file `allow_ram_spill` key is rejected by Config.load() as unknown.
    # `campaign run` routes through cmd_run, so this is the single propagation
    # point for both surfaces (§19).
    cfg.allow_ram_spill = bool(getattr(args, "allow_ram_spill", False))
    # Anvil Stage 1.3: detect GPU inventory exactly once for this invocation
    # and thread it through both backend selection and runtime-identity
    # collection below, rather than letting each independently re-detect.
    inventory = detect_gpus()
    # Anvil Stage 3B.3D: for a real (non-mock) run the client is built against
    # the endpoint the Stage 3B.2 resolver + Stage 3B.3C materialiser verify --
    # reusing a healthy external runtime or launching an owned ephemeral
    # llama-server -- never a stale configured endpoint. A non-RESOLVED
    # resolution or a materialisation failure is a structured pre-row refusal
    # (SystemExit, zero benchmark rows), never a model-quality failure.
    # --mock keeps the offline stub and touches neither the resolver nor the
    # materialiser.
    out_dir = _run_dir(args)
    materialisation = None
    if not getattr(args, "mock", False):
        materialisation = _resolve_and_materialise_for_run(args, cfg, inventory=inventory)
        # A refusal / resolve / materialise failure is a plain pre-row
        # SystemExit: no usable owned runtime was ever handed to us, so there is
        # nothing to clean and no lifecycle scope to enter. (This matches the
        # pre-3B.3D-corrective-3 shape -- only an *ok* outcome opens the scope.)
        if not materialisation.ok:
            # A failed *attempted* materialisation can contain the only audit
            # evidence of candidate ports, argv/CVD, diagnostics and direct
            # child reaping.  Persist it before refusing; a resolver-only
            # refusal still has no materialisation evidence to write.
            if materialisation.materialisation is not None:
                _persist_materialisation_evidence(
                    out_dir, materialisation, benchmark_completed=False,
                    failure_stage="materialisation",
                )
            raise SystemExit(f"run refused: {materialisation.refusal_reason}")
    # Anvil Stage 3B.3D corrective 3 (DEFECT-3B.3D-03): the runtime lifecycle
    # scope opens the moment an owned runtime exists -- before client
    # construction, the runtime-profile endpoint mutation, runtime-identity
    # collection, resume-identity validation, ranking scope, plan construction,
    # the host-code opt-in and the interactive plan confirmation. Client
    # construction can raise (bad endpoint, adapter import failure) *after* the
    # materialiser has already spawned an owned llama-server; the earlier shape
    # built the client between materialisation and `with _lifecycle:`, so that
    # exception leaked the process and skipped the evidence `finally`. Every
    # post-materialisation operation is now inside cleanup scope or the
    # cleanup-protected `finally`.
    _lifecycle = (
        materialisation.controller
        if materialisation is not None and materialisation.controller is not None
        else contextlib.nullcontext()
    )
    _benchmark_completed = False
    _failure_stage: str | None = None
    # Anvil Stage 3B.5 -- inline process RSS observation. Sampled by the CLI
    # (never by materialisation_evidence() itself -- see that function's
    # docstring): once the owned runtime is launch-ready, and again
    # immediately after runner.run() returns/raises, both strictly *inside*
    # `with _lifecycle:` -- __exit__ tears the owned process down on block
    # exit, so a read attempted from the outer `finally` would always be None.
    _rss_launch_ready: int | None = None
    _rss_post_execution: int | None = None
    try:
        with _lifecycle:
            if materialisation is None:
                client = _client(args, cfg, gpu_inventory=inventory)
            else:
                # A failure between here and the first benchmark row is a
                # runtime/client failure, never a model-quality failure -- record
                # the stage so the persisted evidence cannot be misread.
                _failure_stage = "client_construction"
                client = _client_for_materialised_endpoint(
                    materialisation.endpoint, cfg, backend=materialisation.backend
                )
                _rss_launch_ready = _observed_rss_bytes(materialisation)
                # Runtime-identity collection reads args._runtime_profile.endpoint;
                # for a managed (owned) llama-server that endpoint is the
                # *materialised* one, not the resolver's pre-launch endpoint, so
                # identity evidence and the client agree on where the runtime is.
                _existing = getattr(args, "_runtime_profile", None)
                if _existing is not None and _existing.endpoint != materialisation.endpoint:
                    from dataclasses import replace as _dc_replace
                    try:
                        setattr(args, "_runtime_profile",
                                _dc_replace(_existing, endpoint=materialisation.endpoint))
                    except TypeError:
                        pass  # not a dataclass profile -- leave as-is
                _apply_managed_owned_model_selection(args, client, materialisation)
                _failure_stage = "pre_run_gates"
            rankings_dir = _ranking_dir_for(args, run_id=out_dir.name)
            task_ids = _task_ids(args)
            selected_models = getattr(args, "_selected_models", None)
            if selected_models is None:
                selected_models = _resolve_model_selection(args, client)
            # Read-only selected-runtime metadata, collected per model.  This is
            # passed into runner before it may inspect resumable evidence.
            profile = getattr(args, "_runtime_profile", None) or implicit_ollama_profile(cfg)
            try:
                tag_rows = {str(row.get("name")): row for row in client.tags()}
                current_runtime_identities = {
                    model: collect_runtime_identity(client=client, profile=profile, model_name=model,
                                                    model_row=tag_rows.get(model), config=cfg, inventory=inventory)
                    for model in selected_models
                }
            except Exception as exc:
                raise SystemExit(f"run refused: current_runtime_identity_missing: {exc}") from exc
            # The CLI gate is intentionally before ranking scope, plan
            # construction, --auto probes, plan JSON, confirmation, telemetry,
            # and runner execution.
            if args.resume and (out_dir / "raw_results.jsonl").exists():
                try: validate_resume_runtime_identities(out_dir, current_runtime_identities, selected_models)
                except ValueError as exc: raise SystemExit(f"run refused: {exc}") from None
            _write_run_ranking_scope(out_dir, args, rankings_dir=rankings_dir)
            capability_profiles = getattr(args, "_capability_profiles", None)
            plan = getattr(args, "_accepted_plan", None) or _plan_for_args(
                args, cfg, client, selected_models=selected_models, capability_profiles=capability_profiles,
                runtime_identities=current_runtime_identities,
            )
            _require_host_code_opt_in(args, plan)
            _confirm_plan(args, plan)
            # Past every early-exit gate: from here a failure is the runner's,
            # not a client-construction / pre-run-gate failure.
            _failure_stage = None
            try:
                runner.run(client, cfg, level=args.level, out_dir=out_dir,
                       include=args.include_regex, exclude=args.exclude_regex,
                       skip_offload=args.skip_offload,
                       categories=_categories(args),
                       task_ids=task_ids, task_regex=args.task_regex,
                       family_base_only=args.family_base_only,
                       context_aliases_only=args.context_aliases_only,
                       context_only=args.context_only,
                       resume=args.resume, judge_mode=args.judge, dump_subjective=args.dump_subjective,
                       dump_raw=args.dump_raw,
                       status_interval=args.status_interval, live_ui=args.live_ui,
                       sample_mode=args.sample_mode, fingerprint_enabled=args.fingerprint,
                       selected_models=selected_models,
                       capability_profiles=plan.get("capability_profiles") or capability_profiles,
                       auto_probe=bool(getattr(args, "auto", False)),
                       capture_runtime_telemetry=True,
                       runtime_profile=getattr(args, "_runtime_profile", None),
                       runtime_identity=current_runtime_identities,
                       gpu_inventory=inventory)
                _benchmark_completed = True
            except ValueError as exc:
                raise SystemExit(f"run refused: {exc}")
            except KeyboardInterrupt:
                print(f"\nINTERRUPTED: Ctrl+C received. Partial results are preserved in {out_dir}")
                print("Rebuilding partial reports from raw_results.jsonl...")
                try:
                    report.build(out_dir, cfg)
                    print(f"partial reports -> {out_dir}")
                    if rankings_dir is not None:
                        _update_rankings(out_dir.parent, rankings_dir, quiet=False, include_separate=bool(getattr(args, "separate_ranking", False)), only_run_ids=([out_dir.name] if getattr(args, "separate_ranking", False) else None))
                except Exception as exc:
                    print(f"partial report rebuild failed: {exc}")
                    print(f"You can retry with: llm-modelbench report --out {out_dir}")
                raise SystemExit(130)
            finally:
                # Anvil Stage 3B.5: sample post-execution RSS on every exit
                # from runner.run() (success, ValueError->SystemExit,
                # KeyboardInterrupt->SystemExit(130)) while still inside
                # `with _lifecycle:` -- the owned process is still alive here;
                # it is gone by the time the outer `finally` persists evidence.
                _rss_post_execution = _observed_rss_bytes(materialisation)
            report.build(out_dir, cfg)
            validity = runner.assess_run_validity(out_dir)
            print(f"\ndone -> {out_dir}  validity={validity['status']}")
            if validity["status"] == "invalid":
                raise SystemExit("run completed without usable benchmark evidence; reports were preserved, rankings were not updated")
            if getattr(args, "strict_harness", False) and validity["harness_error_rows"]:
                raise SystemExit(
                    f"strict harness check failed: {validity['harness_error_rows']} harness-error row(s); "
                    "reports were preserved, rankings were not updated"
                )
            if rankings_dir is not None:
                _update_rankings(out_dir.parent, rankings_dir, quiet=False, include_separate=bool(getattr(args, "separate_ranking", False)), only_run_ids=([out_dir.name] if getattr(args, "separate_ranking", False) else None))
    finally:
        if materialisation is not None:
            _persist_materialisation_evidence(
                out_dir, materialisation, benchmark_completed=_benchmark_completed,
                failure_stage=_failure_stage,
                rss_bytes_launch_ready=_rss_launch_ready,
                rss_bytes_post_execution=_rss_post_execution,
            )



def cmd_watch(args, cfg):
    from . import watch
    single_run_requested = bool(args.run_id or args.out or args.once)
    follow_queue = args.follow_queue if args.follow_queue is not None else not single_run_requested
    if follow_queue:
        runs_dir = Path(args.runs_dir or "runs")
        return watch.watch_queue(runs_dir, layout=args.layout, refresh=args.refresh,
                                  clear=not args.no_clear, screen=args.screen,
                                  idle_grace_seconds=args.idle_grace)
    if args.run_id:
        run_dir = _run_dir(args)
    elif args.out:
        run_dir = Path(args.out)
    else:
        run_dir = watch.resolve_run_dir(Path(args.runs_dir or "runs"))
    return watch.watch(run_dir, layout=args.layout, refresh=args.refresh,
                       clear=not args.no_clear, once=args.once, screen=args.screen,
                       exit_when_done=bool(getattr(args, "exit_when_done", False)))


def cmd_simulate(args, cfg):
    from . import simulate
    if getattr(args, "simulate_cmd", None) == "repair-watch":
        from .watch_fixtures import replay_repair_watch
        result = replay_repair_watch(
            Path(args.runs_dir or "runs"),
            scenario=args.scenario,
            speed=args.speed,
            run_id=args.run_id,
            render=not args.write_only,
            screen=args.screen,
            keep=not args.cleanup,
        )
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        elif args.write_only:
            print(f"repair-watch fixture -> {result['campaign_dir']}")
            print(f"watch with: ./llmb-watch --run-id {result['campaign_run_id']} --runs-dir {args.runs_dir or 'runs'}")
        else:
            print(f"\nrepair-watch replay complete -> {result['campaign_dir']}")
        return
    if not getattr(args, "run_dir", None) or getattr(args, "simulate_vram", None) is None:
        raise SystemExit(
            "simulate requires either 'repair-watch' or legacy --run-dir and --simulate-vram arguments"
        )
    rows = simulate.load_rows(Path(args.run_dir))
    results = simulate.simulate(rows, args.simulate_vram)
    print(json.dumps(results, indent=2) if args.json else simulate.report(results, args.simulate_vram))


def cmd_context_profile(args, cfg):
    from .context_profile import run_context_profile
    client = _client(args, cfg)
    run_id = args.run_id or datetime.now().strftime("context_profile_%Y%m%d_%H%M%S")
    run_dir = Path(args.runs_dir or "runs") / run_id
    rankings_dir = _ranking_dir_for(args, run_id=run_id)
    _confirm_destructive_compute(
        f"Run one controlled long-context telemetry profile for {args.model} up to {args.target_ctx} tokens?",
        yes=bool(args.yes),
    )
    try:
        result = run_context_profile(
            client, cfg,
            model=args.model,
            run_dir=run_dir,
            rankings_dir=rankings_dir,
            cards_dir=(Path(args.cards_out) if (args.cards_out and rankings_dir is not None) else None),
            target_ctx=args.target_ctx,
            gpu_vram_gb=args.gpu_vram_gb,
            emergency_headroom_gb=args.emergency_headroom_gb,
            max_spill_gb=args.max_spill_gb,
            min_tps=args.min_tps,
            critical_tps=args.critical_tps,
            live_ui=args.live_ui,
            behavior_probe=bool(args.behavior_probe),
            ranking_scope=("separate" if getattr(args, "separate_ranking", False) else "canonical"),
        )
    except (FileExistsError, ValueError) as exc:
        raise SystemExit(f"context-profile refused: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    if not (result.get("telemetry_validation") or {}).get("passed"):
        raise SystemExit(3)


def cmd_model_cards(args, cfg):
    from .model_cards import generate_model_cards
    result = generate_model_cards(
        Path(args.rankings_dir), Path(args.out),
        runs_dir=(Path(args.runs_dir) if args.runs_dir else None),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_freeze(args, cfg):
    from .freeze import create_freeze, verify_freeze
    if args.verify:
        result = verify_freeze(Path(args.out))
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result.get("passed"):
            raise SystemExit(4)
        return
    result = create_freeze(
        Path(args.repo_root), Path(args.runs_dir), Path(args.rankings_dir), Path(args.out),
        label=args.label, include_rankings=not args.no_rankings_copy,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_serve(args, cfg):
    from . import serve
    try:
        serve.serve(
            [Path(path) for path in args.runs_dir], args.host, args.port,
            allow_remote=bool(args.allow_remote), allow_empty=bool(args.allow_empty),
        )
    except ValueError as exc:
        raise SystemExit(f"serve refused: {exc}") from exc


def cmd_report(args, cfg):
    from . import report
    run_dir = _require_run_dir(args, command="report")
    if args.weights:
        from .weights_override import copy_run_for_override, parse_weight_overrides
        cfg.weights = parse_weight_overrides(args.weights, cfg.weights)
        cfg.weight_override_spec = args.weights
        override_out = Path(args.report_out) if args.report_out else run_dir.parent / f"{run_dir.name}_weight_override"
        run_dir = copy_run_for_override(run_dir, override_out)
    report.build(run_dir, cfg)


def cmd_pack(args, cfg):
    from . import runner
    runner.pack_subjective(_require_run_dir(args, command="pack-subjective"))


def cmd_doctor(args, cfg):
    from . import doctor
    data = doctor.collect(cfg)
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print(doctor.render(data))


def cmd_plan(args, cfg):
    from . import planner
    if args.family_base_only and args.context_aliases_only:
        raise SystemExit("--family-base-only and --context-aliases-only cannot be used together")
    if args.samples is not None:
        cfg.samples = args.samples
    if getattr(args, "ctx", None):
        cfg.ctx_override = int(args.ctx)
    if getattr(args, "num_predict", None):
        cfg.num_predict_override = int(args.num_predict)
    if getattr(args, "think", None):
        cfg.think = args.think
    if getattr(args, "needle_max_ctx", None):
        cfg.needle_max_ctx = int(args.needle_max_ctx)
    if getattr(args, "judge_model", None):
        cfg.judge_model = args.judge_model
    client = _client(args, cfg)
    selected_models = _resolve_model_selection(args, client)
    plan = _plan_for_args(args, cfg, client, selected_models=selected_models)
    if args.json:
        print(json.dumps(plan, indent=2))
    else:
        print(planner.render_plan(plan))
    if args.plan_json:
        planner.write_plan(Path(args.plan_json), plan)
        print(f"plan json -> {args.plan_json}")


def _campaign_paths_or_exit(campaign_id: str):
    from . import campaign
    paths = campaign.resolve_paths(campaign_id)
    if not paths.manifest.exists():
        raise SystemExit(f"unknown campaign {campaign_id!r}")
    return paths, campaign.load_manifest(paths)

def _campaign_runtime_identities(args, cfg, selected, client, *, gpu_inventory=None):
    """Bounded metadata only; never probes or generates.

    ``gpu_inventory``, when supplied by a caller that already detected GPUs
    for this invocation (Anvil Stage 1.3), is reused here instead of a
    second ``detect_gpus()`` call."""
    from .hardware import detect_gpus
    from .runtime_identity import collect_runtime_identity
    profile=getattr(args,"_runtime_profile",None) or implicit_ollama_profile(cfg)
    rows={str(x.get("name")):x for x in client.tags()}
    inventory=tuple(gpu_inventory) if gpu_inventory is not None else detect_gpus()
    return {model:collect_runtime_identity(client=client,profile=profile,model_name=model,model_row=rows.get(model),inventory=inventory,config=cfg) for model in selected}

def _validate_campaign_generation_identities(paths, args, cfg, manifest):
    """Called before interrupted state transition/lock; recovering shares generation cohort."""
    from .runtime_identity import validate_frozen_runtime_identity_map
    try: plan=json.loads(paths.plan_json.read_text(encoding="utf-8"))
    except Exception: raise SystemExit("campaign resume refused: runtime identity mismatch: runtime_identity_artifact_unavailable")
    if manifest.resume_state == "judging":
        judge=plan.get("judge_runtime_identity")
        if not isinstance(judge,dict) or judge.get("state") in {None,"judge_not_required"}: raise SystemExit("campaign resume refused: judge_runtime_identity_missing")
    frozen=plan.get("runtime_identities")
    if frozen is None: raise SystemExit("campaign resume refused: runtime identity mismatch: legacy_runtime_identity_missing")
    from .hardware import detect_gpus
    inventory=detect_gpus()
    client=_client(args,cfg,gpu_inventory=inventory); selected=[str(x) for x in frozen]
    try: validate_frozen_runtime_identity_map(frozen,_campaign_runtime_identities(args,cfg,selected,client,gpu_inventory=inventory),selected)
    except ValueError as exc: raise SystemExit("campaign resume refused: "+str(exc)) from None
    return client, selected


def cmd_campaign(args, cfg):
    """Thin compatibility layer: existing runners receive a normal nested run dir."""
    from . import campaign
    if args.campaign_cmd == "init":
        destination = Path(args.path)
        if destination.exists():
            raise SystemExit(f"campaign init refused: {destination} already exists")
        template = campaign.campaign_config_template()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
        print(f"campaign template -> {destination}")
        return
    if args.campaign_cmd == "execute":
        source = Path(args.campaign_config)
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"campaign execute requires JSON campaign config: {exc}") from None
        try:
            data = campaign.validate_campaign_config(data)
        except campaign.CampaignError as exc:
            raise SystemExit(f"campaign execute refused: {exc}") from None
        campaign_id = str(data["campaign_id"])
        models = data["models"]
        paths = campaign.resolve_paths(campaign_id)
        config_record = campaign.campaign_config_plan_record(data)
        config_plan_path = paths.plan_dir / "campaign_config.json"
        if paths.manifest.exists():
            manifest = campaign.load_manifest(paths)
            if not config_plan_path.exists():
                raise SystemExit(
                    "campaign execute refused: existing campaign lacks immutable config plan; "
                    f"use explicit campaign commands or inspect {paths.root}"
                )
            try:
                existing_record = json.loads(config_plan_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise SystemExit(f"campaign execute refused: invalid immutable config plan: {exc}") from None
            if existing_record.get("config_signature") != config_record["config_signature"]:
                raise SystemExit("campaign execute refused: existing campaign was planned with different config")
            if manifest.state == "packaged":
                print(json.dumps({
                    "campaign_id": campaign_id,
                    "state": manifest.state,
                    "result": "noop",
                    "message": "identical config already completed; no evidence changed",
                }, indent=2, sort_keys=True))
                return
            if manifest.state == "interrupted":
                raise SystemExit(
                    f"campaign execute paused: campaign is interrupted; valid next action is campaign resume {campaign_id}"
                )
            raise SystemExit(
                f"campaign execute refused: campaign state {manifest.state!r} is not safe for config execution; "
                f"valid next action is campaign status {campaign_id}"
            )
        paths, manifest = campaign.create_campaign(
            campaign_id,
            models=list(map(str, models)),
            level=str(data.get("level") or "full"),
            version=__version__,
        )
        manifest.notes["campaign_config"] = {
            "managed": True,
            "schema_version": campaign.CAMPAIGN_CONFIG_SCHEMA_VERSION,
            "config_signature": config_record["config_signature"],
            "path": "plan/campaign_config.json",
        }
        campaign.write_manifest(paths, manifest)
        campaign.transition(paths, manifest, "planned")
        campaign._atomic_write_text(config_plan_path, json.dumps(config_record, indent=2, sort_keys=True))
        try:
            frozen_record = json.loads(config_plan_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"campaign execute refused: immutable config plan verification failed: {exc}") from None
        if frozen_record != config_record:
            raise SystemExit("campaign execute refused: immutable config plan verification failed")
        invocation = ["campaign", "run", "--campaign-id", campaign_id, "--models", ";".join(map(str, models)),
                      "--level", str(data.get("level") or "full"), "--yes", "--unattended-safe"]
        if data.get("runtime_policy", {}).get("auto", True): invocation.append("--auto")
        if data.get("samples") is not None: invocation += ["--samples", str(data["samples"])]
        needle = (data.get("context_needle_policy") or {}).get("needle_max_ctx")
        if needle is not None: invocation += ["--needle-max-ctx", str(needle)]
        if (data.get("executable_scorer_policy") or {}).get("allow_host_code_execution"):
            invocation.append("--allow-host-code-execution")
        if args.mock: invocation.append("--mock")
        # Normal lifecycle remains campaign-owned and intentionally stops before adoption.
        main(invocation)
        return
    if args.campaign_cmd == "supersede":
        paths, _ = _campaign_paths_or_exit(args.campaign_id)
        try:
            replacement = json.loads(Path(args.replacement_row).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"campaign supersede requires JSON replacement row file: {exc}") from None
        try:
            if args.source_row_hash:
                source_hash = str(args.source_row_hash)
            elif args.source_row:
                source_candidate = json.loads(Path(args.source_row).read_text(encoding="utf-8"))
                source_hash = campaign._primary_row_hash(source_candidate)
            else:
                raise campaign.CampaignError("source_row_hash_required")
            source = campaign.primary_row_by_hash(paths, source_hash)
            replacement_preview_hash = campaign._primary_row_hash(replacement)
            replacement_hash = str(args.replacement_row_hash or replacement_preview_hash)
            if args.dry_run and args.replacement_row_hash and str(args.replacement_row_hash) != replacement_preview_hash:
                raise campaign.CampaignError("replacement_row_hash_mismatch")
            item = campaign._native_supersession_record(
                paths=paths,
                source_campaign_id=args.source_campaign_id or args.campaign_id,
                source_run_id=args.source_run_id,
                source_row=source,
                replacement_campaign_id=args.replacement_campaign_id or args.campaign_id,
                replacement_run_id=args.replacement_run_id,
                replacement_row=replacement,
                reason=args.reason,
                operator=args.operator,
                tool="llmb",
            )
            existing_edges = campaign.load_supersession_ledger(paths)
            candidate_edge = campaign.validate_supersession_record(item, source_row=source, replacement_row=replacement)
            graph = campaign.build_supersession_graph(existing_edges + [candidate_edge])
            if not graph["valid"]:
                reason = str((graph["errors"] or [{}])[0].get("reason") or "invalid_supersession_graph")
                raise campaign.CampaignError(reason)
            if args.dry_run:
                print(json.dumps({"dry_run": True, "would_record": item}, indent=2, sort_keys=True))
            else:
                stored_replacement = campaign.stored_replacement_row_by_hash(
                    campaign_id=args.replacement_campaign_id or args.campaign_id,
                    run_id=args.replacement_run_id,
                    row_hash=replacement_hash,
                )
                if stored_replacement != replacement:
                    raise campaign.CampaignError("replacement_preview_contradicts_stored_evidence")
                campaign.validate_replacement_provenance_matches_source(source, stored_replacement)
                item = campaign.record_supersession(
                    paths,
                    source_campaign_id=args.source_campaign_id or args.campaign_id,
                    source_run_id=args.source_run_id,
                    source_row=source,
                    replacement_campaign_id=args.replacement_campaign_id or args.campaign_id,
                    replacement_run_id=args.replacement_run_id,
                    replacement_row=stored_replacement,
                    reason=args.reason,
                    operator=args.operator,
                    tool="llmb",
                )
                print(json.dumps(item, indent=2, sort_keys=True))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"campaign supersede requires JSON source row file: {exc}") from None
        except campaign.CampaignError as exc:
            raise SystemExit(f"campaign supersede refused: {exc}") from None
        return
    if args.campaign_cmd == "status":
        paths, manifest = _campaign_paths_or_exit(args.campaign_id)
        print(json.dumps({"campaign_id": manifest.campaign_id, "state": manifest.state,
                          "resume_state": manifest.resume_state, "root": str(paths.root)}, indent=2))
        return
    if args.campaign_cmd == "resume":
        paths, manifest = _campaign_paths_or_exit(args.campaign_id)
        if manifest.state != "interrupted" or not manifest.resume_state:
            raise SystemExit("campaign resume requires an interrupted campaign with a recorded resume phase")
        _validate_campaign_generation_identities(paths,args,cfg,manifest)
        resumed = campaign.transition(paths, manifest, manifest.resume_state)
        print(json.dumps({"campaign_id": resumed.campaign_id, "state": resumed.state,
                          "resumed_phase": resumed.state, "root": str(paths.root)}, indent=2))
        return
    if args.campaign_cmd == "package":
        paths, _ = _campaign_paths_or_exit(args.campaign_id)
        package = campaign.package_campaign(paths)
        print(f"campaign package -> {package}")
        return
    if args.campaign_cmd == "clean":
        if args.all and args.campaign_id:
            raise SystemExit("campaign clean accepts either campaign_id or --all, not both")
        if args.all:
            result = campaign.cleanup_all_campaigns(apply=bool(args.apply))
        else:
            if not args.campaign_id:
                raise SystemExit("campaign clean requires a campaign_id or --all")
            paths, _ = _campaign_paths_or_exit(args.campaign_id)
            result = campaign.cleanup_campaign(paths, apply=bool(args.apply))
        print(json.dumps(result, indent=2))
        return
    if args.campaign_cmd == "migrate-legacy":
        result = campaign.migrate_legacy_run(args.run_id, args.campaign_id, runs_dir=Path(args.runs_dir), apply=bool(args.apply))
        print(json.dumps(result, indent=2))
        return
    if args.campaign_cmd == "plan":
        paths = campaign.resolve_paths(args.campaign_id)
        if paths.manifest.exists():
            manifest = campaign.load_manifest(paths)
            if manifest.state not in {"created", "planned"}:
                raise SystemExit(campaign.campaign_replan_refusal(manifest))
        else:
            models = [value.strip() for value in (args.models or "").split(";") if value.strip()]
            paths, manifest = campaign.create_campaign(args.campaign_id, models=models, level=args.level, version=__version__)
        from .hardware import detect_gpus
        inventory = detect_gpus()
        client = _client(args, cfg, gpu_inventory=inventory)
        selected = _resolve_model_selection(args, client) or []
        identities=_campaign_runtime_identities(args,cfg,selected,client,gpu_inventory=inventory)
        plan = _plan_for_args(args, cfg, client, selected_models=selected, runtime_identities=identities)
        configuration = {"level": args.level, "models": args.models, "judge_policy": getattr(args, "judge", "off"), "samples": args.samples, "think": args.think, "ctx": args.ctx, "num_predict": args.num_predict}
        if manifest.state == "planned":
            if not paths.plan_json.exists():
                raise SystemExit(campaign.campaign_replan_refusal(manifest))
            existing = json.loads(paths.plan_json.read_text(encoding="utf-8"))
            proposed = campaign._campaign_plan_payload(paths, plan, configuration=configuration, created_at=existing.get("created_at"), runtime_identities={k:v.to_dict() for k,v in identities.items()})
            if campaign.campaign_plan_equivalent(existing, proposed):
                print(json.dumps({
                    "campaign_id": manifest.campaign_id,
                    "state": manifest.state,
                    "result": "noop",
                    "plan": str(paths.plan_json),
                    "message": "identical campaign plan already persisted; no files changed",
                }, indent=2))
                return
            raise SystemExit(campaign.campaign_replan_refusal(manifest))
        campaign.write_campaign_plan(paths, plan, inventory=client.tags(), capabilities=plan.get("capability_profiles") or {}, configuration=configuration, runtime_identities={k:v.to_dict() for k,v in identities.items()})
        if manifest.state == "created":
            campaign.transition(paths, manifest, "planned")
        print(f"campaign plan -> {paths.plan_json}")
        return
    if args.campaign_cmd == "run":
        paths = campaign.resolve_paths(args.campaign_id)
        if not paths.manifest.exists():
            models = [value.strip() for value in (args.models or "").split(";") if value.strip()]
            paths, manifest = campaign.create_campaign(args.campaign_id, models=models, level=args.level, version=__version__)
            manifest = campaign.transition(paths, manifest, "planned")
        else:
            manifest = campaign.load_manifest(paths)
        if manifest.state == "planned":
            manifest = campaign.transition(paths, manifest, "generating")
        elif manifest.state == "interrupted" and manifest.resume_state == "generating":
            _validate_campaign_generation_identities(paths,args,cfg,manifest)
            manifest = campaign.transition(paths, manifest, "generating")
        elif manifest.state != "generating":
            raise SystemExit(f"campaign {args.campaign_id!r} cannot run from state {manifest.state!r}")
        lock = campaign.acquire_lock(paths, operation="campaign-run", phase="generating")
        try:
            client = _client(args, cfg)
            selected = _resolve_model_selection(args, client)
            # Anvil Stage 3.6: thread runtime identities so the accepted
            # campaign plan resolves the active protocol and canonical
            # benchmark runtime instead of deferring it (section 30). When
            # no explicit selection filter is given, the campaign manifest
            # is the model set.
            _identity_models = selected or list(manifest.models)
            try:
                _run_identities = (
                    _campaign_runtime_identities(args, cfg, _identity_models, client)
                    if _identity_models else {}
                )
            except Exception:
                # D7: the prior-knowledge surface must never introduce a new
                # "run refused" failure mode. Canonical benchmark runtime
                # falls back to deferred; the run proceeds.
                _run_identities = {}
            accepted_plan = _plan_for_args(
                args, cfg, client, selected_models=selected,
                runtime_identities=_run_identities,
            )
            campaign.write_campaign_plan(paths, accepted_plan, inventory=client.tags(), capabilities=accepted_plan.get("capability_profiles") or {}, configuration={"level": args.level, "models": args.models, "judge_policy": getattr(args, "judge", "off"), "samples": args.samples, "think": args.think, "ctx": args.ctx, "num_predict": args.num_predict})
            args._accepted_plan = accepted_plan
            args._selected_models = selected
            args.judge = "off"
            args.out = str(paths.evidence_dir)
            args.run_id = "primary"
            args.rankings_out = str(paths.candidate_rankings_dir)
            args.separate_ranking = True
            args.no_ranking_update = False
            cmd_run(args, cfg)
            campaign.sync_primary_reports(paths)
            rows = [json.loads(line) for line in paths.primary_raw_results.read_text(encoding="utf-8").splitlines() if line.strip()]
            identities = json.loads((paths.primary_dir / "model_identities.json").read_text(encoding="utf-8")) if (paths.primary_dir / "model_identities.json").exists() else {}
            for row in rows:
                identity = identities.get(row.get("model")) or {}
                row["model_digest_resolved"] = identity.get("digest") or row.get("model_digest") or row.get("model")
            retry_rows = [row for row in rows if campaign.classify_recovery_row(row)["retry"]]
            if retry_rows:
                # Execute the existing bounded repair engine against nested primary evidence.
                manifest_now = campaign.load_manifest(paths)
                result = campaign.execute_recovery_phase(paths, client, cfg, budget=int(args.num_predict or 2048))
            # Primary generation is always judge-off. Subjective judging is post-hoc.
            from .tasks import TASKS
            subjective = {task.id for task in TASKS if task.scorer == "subjective"}
            eligible = [row for row in rows if row.get("task") in subjective and not row.get("error_kind")]
            if eligible:
                manifest_now = campaign.load_manifest(paths)
                if manifest_now.state in {"generating", "recovering"}:
                    campaign.transition(paths, manifest_now, "judging")
                inventory = client.tags()
                cohort_by_key = {}
                for row in rows:
                    key = (str(row.get("model") or ""), str(row.get("model_digest_resolved") or ""))
                    cohort_by_key.setdefault(key, {"name": row.get("model"), "digest": row.get("model_digest_resolved")})
                cohort = list(cohort_by_key.values())
                judge_policy = campaign.JudgePolicy.from_config(cfg, enabled=True)
                plan_profiles = accepted_plan.get("capability_profiles") or {}
                candidates = []
                for item in inventory:
                    name = str(item.get("name") or "")
                    profile = dict(plan_profiles.get(name) or {})
                    candidate = {
                        **profile,
                        "name": name,
                        "digest": item.get("digest"),
                        "capabilities": client.capability_hints(name),
                        "priority": 0,
                        "calibrated": False,
                    }
                    candidates.append(candidate)
                candidates = campaign.apply_campaign_roles_to_judge_candidates(candidates, cohort, judge_policy)
                # Anvil Stage 3.5: the judge capability-eligibility gate prefers
                # native identity-compatible EvidenceLedger evidence over the
                # legacy adapter. Campaign capability evidence lives at one
                # canonical path under the campaign evidence dir.
                capability_ledger = campaign._campaign_capability_ledger(paths.evidence_dir)
                judge_selection = campaign.build_judge_selection(candidates, cohort, judge_policy, ledger=capability_ledger)
                qualified_judges, qualifications, coverage = campaign.select_qualified_campaign_judges_for_rows(client, judge_selection, eligible, ledger=capability_ledger)
                judge = qualified_judges[0] if qualified_judges else None
                qualification = (judge or {}).get("qualification") if judge else (qualifications[-1] if qualifications else None)
                selection = {"eligible": len(eligible), "cohort": cohort, "machine_judged_provisional": True, "judge": judge,
                             "qualified_judges": qualified_judges, "qualification": qualification, "qualification_chain": qualifications, "qualification_coverage": coverage,
                             "posthoc_judge_model": (judge or {}).get("name"), "posthoc_judge_digest": (judge or {}).get("digest"), "generation_judge_model": None,
                             "model_role_policy_version": campaign.MODEL_ROLE_POLICY_VERSION,
                             "judge_policy_version": campaign.JUDGE_POLICY_VERSION,
                             "judge_policy_selection": judge_selection.to_dict()}
                campaign._atomic_write_text(paths.judge_dir / "judge_selection.json", json.dumps(selection, indent=2, sort_keys=True))
                if judge:
                    from . import judge_dumps
                    judged, selection = campaign.judge_run_with_structural_continuation(
                        client,
                        paths.primary_dir,
                        selection=judge_selection,
                        selection_evidence=selection,
                        qualified_judges=qualified_judges,
                        qualifications=qualifications,
                        source_rows=eligible,
                        judge_model=judge["name"],
                        judge_mode="single",
                    )
                    campaign._atomic_write_text(paths.judge_dir / "judge_selection.json", json.dumps(selection, indent=2, sort_keys=True))
                    if (paths.primary_dir / "judge_results.jsonl").exists():
                        __import__("shutil").copy2(paths.primary_dir / "judge_results.jsonl", paths.judge_results)
                    campaign._atomic_write_text(paths.judge_summary, json.dumps({**judged, "selection": selection}, indent=2, sort_keys=True))
                    raw_rows = [json.loads(line) for line in paths.primary_raw_results.read_text(encoding="utf-8").splitlines() if line.strip()]
                    rows = judge_dumps.apply_judgements(paths.primary_dir, raw_rows)
                    for row in rows:
                        identity = identities.get(row.get("model")) or {}
                        row["model_digest_resolved"] = identity.get("digest") or row.get("model_digest") or row.get("model")
                else:
                    campaign._atomic_write_text(paths.judge_summary, json.dumps({"status": "awaiting_external_judge", "selection": selection}, indent=2, sort_keys=True))
                campaign.transition(paths, campaign.load_manifest(paths), "packaged")
            elif campaign.load_manifest(paths).state in {"generating", "recovering"}:
                campaign.transition(paths, campaign.load_manifest(paths), "packaged")
            for row in rows:
                row["disposition"] = campaign.classify_recovery_row(row)["disposition"]
            campaign.write_readiness(paths, rows, judge_available=True)
            if getattr(args, "unattended_safe", False):
                campaign.package_campaign(paths, allow_active_lock=True)
        except KeyboardInterrupt:
            campaign.transition(paths, campaign.load_manifest(paths), "interrupted")
            raise
        finally:
            campaign.release_lock(paths, lock)
        return
    raise SystemExit("campaign command required")


def cmd_grade(args, cfg):
    from . import grade
    run_dir = _require_run_dir(args, command="grade")
    if args.export_blind:
        pack = grade.export_blind(run_dir)
        print(f"blind pack -> {pack}")
        print(f"mapping -> {pack.parent / 'blind_mapping.json'}")
    else:
        grade.interactive_grade(run_dir)


def cmd_judge_dumps(args, cfg):
    from . import judge_dumps, report

    if getattr(args, "judge_model", None):
        cfg.judge_model = args.judge_model
    if getattr(args, "ctx", None):
        cfg.ctx_override = int(args.ctx)
    if getattr(args, "think", None):
        cfg.think = args.think
    if not str(getattr(cfg, "judge_model", "") or "").strip():
        raise SystemExit("judge-dumps requires --judge-model or configured judge_model")
    client = _client(args, cfg)

    try:
        _cmd_judge_dumps_run(args, cfg, client, judge_dumps, report)
    except judge_dumps.ManualJudgeIneligibleError as exc:
        raise SystemExit(str(exc)) from exc


def _cmd_judge_dumps_run(args, cfg, client, judge_dumps, report):
    if args.everything:
        runs_dir = Path(args.runs_dir or "runs")
        rankings_dir = _ranking_dir_for(args, run_id="judge_everything")
        preview = judge_dumps.judge_everything(
            client, runs_dir, judge_model=cfg.judge_model, judge_mode=args.judge,
            num_ctx=cfg.ctx_override, think=cfg.think, dry_run=True, force=args.force,
        )
        print(f"judge-dumps scan: {preview['runs_scanned']} runs, {preview['eligible']} eligible subjective rows, "
              f"{preview['skipped']} skipped/already judged")
        if args.dry_run:
            print(json.dumps(preview, indent=2))
            return
        _confirm_destructive_compute(
            f"Run {args.judge} post-hoc judging with {cfg.judge_model!r} over {preview['eligible']} eligible row(s)?",
            yes=args.yes,
        )
        result = judge_dumps.judge_everything(
            client, runs_dir, judge_model=cfg.judge_model, judge_mode=args.judge,
            num_ctx=cfg.ctx_override, think=cfg.think, dry_run=False, force=args.force,
            progress=lambda index, total, run, item: print(
                f"[{index}/{total}] {run.name}: eligible={item.get('eligible', 0)} "
                f"judged={item.get('judged', 0)} errors={item.get('judge_errors', 0)}"
            ),
        )
        for item in result["runs"]:
            if item.get("written"):
                run_dir = Path(item["run_dir"])
                report.build(run_dir, cfg)
        print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=2))
        if rankings_dir is not None:
            _update_rankings(runs_dir, rankings_dir, quiet=False, force=True, include_separate=bool(getattr(args, "separate_ranking", False)))
        return

    run_dir = _require_run_dir(args, command="judge-dumps")
    rankings_dir = _ranking_dir_for(args, run_id=run_dir.name)
    preview = judge_dumps.judge_run(
        client, run_dir, judge_model=cfg.judge_model, judge_mode=args.judge,
        num_ctx=cfg.ctx_override, think=cfg.think, dry_run=True, force=args.force,
    )
    print(f"judge-dumps scan: run={run_dir.name} eligible={preview['eligible']} skipped={len(preview['skipped'])}")
    if args.dry_run:
        print(json.dumps(preview, indent=2))
        return
    _confirm_destructive_compute(
        f"Run {args.judge} post-hoc judging with {cfg.judge_model!r} over {preview['eligible']} eligible row(s)?",
        yes=args.yes,
    )
    result = judge_dumps.judge_run(
        client, run_dir, judge_model=cfg.judge_model, judge_mode=args.judge,
        num_ctx=cfg.ctx_override, think=cfg.think, dry_run=False, force=args.force,
    )
    if result.get("written"):
        report.build(run_dir, cfg)
    print(json.dumps({k: v for k, v in result.items() if k != "entries"}, indent=2))
    if rankings_dir is not None:
        _update_rankings(run_dir.parent, rankings_dir, quiet=False, force=True, include_separate=bool(getattr(args, "separate_ranking", False)), only_run_ids=([run_dir.name] if getattr(args, "separate_ranking", False) else None))


def cmd_repair(args, cfg):
    from . import repair

    if args.kv_cascade and not args.restart_ollama:
        raise SystemExit("--kv-cascade requires --restart-ollama")
    if args.restart_ollama and not args.kv_cascade:
        raise SystemExit("--restart-ollama is only valid with --kv-cascade")
    if args.restart_ollama and args.mock:
        raise SystemExit("--restart-ollama cannot be combined with --mock")
    if args.restart_ollama and not args.apply:
        # Planning is still safe and useful; the flag describes the intended
        # apply mode and does not touch systemd during dry-run.
        pass
    if args.kv_cascade and args.kv_type != "current":
        raise SystemExit("--kv-cascade owns the q8_0 -> q4_0 sequence; leave --kv-type at current")
    if args.keep_final_kv and not args.kv_cascade:
        raise SystemExit("--keep-final-kv is only valid with --kv-cascade")
    auto_confirm = bool(getattr(args, "auto_confirm", False))
    if auto_confirm and not args.restart_ollama:
        raise SystemExit("--auto-confirm is only valid with --restart-ollama --kv-cascade")

    if getattr(args, "judge_model", None):
        cfg.judge_model = args.judge_model
    plan = repair.build_plan(
        Path(args.runs_dir or "runs"),
        run_id=args.run_id, run_prefix=args.run_prefix, everything=args.everything,
        think_retry_num_predict=args.num_predict,
        retry_transient=not args.no_transient_retry,
        include_missing=not args.no_missing_tasks,
        judge_mode=args.judge,
        judge_model=(cfg.judge_model if args.judge != "off" else None),
        emergency_headroom_gb=args.emergency_headroom_gb,
        max_spill_gb=args.max_spill_gb,
        kv_type=("current" if args.kv_cascade else args.kv_type),
        kv_server_confirmed=args.confirm_kv_server,
        gpu_total_gb=args.gpu_vram_gb,
        force=args.force,
    )
    print(repair.render_plan(plan))
    if args.kv_cascade:
        needle_count = sum(1 for action in plan.actions if action.kind == "retry_needle_guarded")
        print("\nUnattended current-first KV repair" if auto_confirm else "\nCurrent-first managed KV repair")
        service_label = (
            f"auto-discover owner of {cfg.ollama_url}"
            if args.ollama_service == "auto" else args.ollama_service
        )
        print(f"  service fallback target: {service_label}")
        print("  phases: current/default KV -> unresolved-only q8_0 -> unresolved-only q4_0 -> restore if mutated")
        print(f"  guarded needle actions: {needle_count}")
        print("  sudo/service discovery: deferred until current/default KV leaves work unresolved")
        if auto_confirm:
            print("  typed confirmations: skipped; fallback privileged commands use sudo -n only")
        else:
            print("  fallback privileged phases require typed confirmation; sudo owns the password prompt")
        if args.keep_final_kv:
            print("  WARNING: --keep-final-kv leaves Ollama at the final fallback setting")
    plan_path = Path(args.plan_out or (Path(args.runs_dir or "runs") / f"repair_plan_{plan.plan_id}.json"))
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    print(f"repair plan -> {plan_path}")
    if not args.apply:
        print("dry-run only; no model calls, judgements, or source evidence changes were made")
        return
    if not plan.actions:
        print("nothing to apply: the plan contains no automatic repair actions")
        print("If a previous unresolved repair is intentionally being repeated, rerun with --force.")
        return
    _confirm_destructive_compute(
        f"Apply {len(plan.actions)} bounded repair action(s) from plan {plan.plan_id}?",
        yes=bool(args.yes or auto_confirm),
    )
    client = _client(args, cfg)
    if args.kv_cascade:
        require_capability(client, BackendCapability.OLLAMA_SERVICE_REPAIR)
        require_capability(client, BackendCapability.OLLAMA_KV_REPAIR)
    rankings_dir = _ranking_dir_for(args, run_id=f"repair_{plan.plan_id}")
    ranking_scope = "separate" if getattr(args, "separate_ranking", False) else "canonical"
    if args.kv_cascade:
        from .ollama_service import BrokerOllamaServiceController
        service_audit_path = Path(args.runs_dir or "runs") / f"repair_service_{plan.plan_id}.jsonl"
        controller_holder = {"controller": None}

        def record_service_event(event):
            entry = {
                "plan_id": plan.plan_id,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                **event,
            }
            with service_audit_path.open("a") as handle:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")

        print(f"service audit -> {service_audit_path}")
        port = _ollama_port(cfg.ollama_url)
        force_password = not args.reuse_sudo_credentials

        def make_controller():
            # Deliberately lazy: current/default KV is attempted before any sudo
            # preflight, service discovery, or systemd mutation. This factory is
            # called only when unresolved needle work genuinely needs q8/q4.
            if controller_holder["controller"] is not None:
                return controller_holder["controller"]
            if auto_confirm:
                print(
                    "\nCurrent/default KV left guarded needle work unresolved. "
                    "Entering unattended quantized-KV fallback. Privileged commands use "
                    "'sudo -n' only; running the scoped NOPASSWD preflight now."
                )
                preflight = BrokerOllamaServiceController(
                    "ollama.service", port=port, auto_confirm=True,
                )
                preflight.verify_noninteractive_sudo_ready()
                print("Preflight passed: passwordless sudo is ready for the fallback phases.\n")
            if args.ollama_service == "auto":
                discovery_guard = BrokerOllamaServiceController(
                    "ollama.service", port=port,
                    force_password_prompt=force_password,
                    auto_confirm=auto_confirm,
                )
                discovery_guard.confirm(
                    "discover",
                    f"LLM ModelBench will identify the systemd service that owns the process "
                    f"listening on {cfg.ollama_url}. No service will be changed in this phase. "
                    + ("sudo -n is used; no password prompt is permitted." if auto_confirm
                       else "sudo may ask for your password."),
                    keyword="DISCOVER",
                )
                discovery_guard.authorise_sudo()
                controller = BrokerOllamaServiceController.for_active_service(
                    port=port, force_password_prompt=force_password,
                    auto_confirm=auto_confirm, event_callback=record_service_event,
                    warn_fn=lambda message: record_service_event({
                        "phase": "discover", "warning": message,
                    }),
                )
            else:
                controller = BrokerOllamaServiceController(
                    args.ollama_service, port=port,
                    force_password_prompt=force_password,
                    auto_confirm=auto_confirm, event_callback=record_service_event,
                )
                controller.confirm(
                    "verify",
                    f"LLM ModelBench will verify that {controller.unit} owns the live Ollama "
                    f"process on {cfg.ollama_url}. No service will be changed in this phase. "
                    + ("sudo -n is used; no password prompt is permitted." if auto_confirm
                       else "sudo may ask for your password."),
                    keyword="VERIFY",
                )
                controller.authorise_sudo()
            active_service = controller.verify_owns_live_process()
            gpu_warning = controller.verify_gpu_binding()
            discovery_event = {
                "phase": "discovery", "unit": active_service.unit,
                "pid": active_service.pid, "port": active_service.port,
                "verified": True,
                "note": "active Ollama service resolved from listener PID and systemd MainPID",
            }
            if gpu_warning:
                discovery_event["warning"] = gpu_warning
            controller.events.append(discovery_event)
            record_service_event(discovery_event)
            print(
                f"active Ollama service -> {active_service.unit} "
                f"(PID {active_service.pid}, port {active_service.port})"
            )
            controller_holder["controller"] = controller
            return controller

        try:
            result = repair.apply_plan_with_managed_kv_cascade(
                client, cfg, plan, None, controller_factory=make_controller,
                auto_confirm=auto_confirm, judge_mode=args.judge,
                judge_model=(cfg.judge_model if args.judge != "off" else None),
                rankings_dir=rankings_dir,
                keep_final_kv=args.keep_final_kv,
                live_ui=args.live_ui,
                ranking_scope=ranking_scope,
            )
            result["service_audit_path"] = str(service_audit_path)
        except Exception as exc:
            controller = controller_holder.get("controller")
            failure = {
                "plan_id": plan.plan_id, "outcome": "FAILED",
                "error": repr(exc), "service_audit_path": str(service_audit_path),
                "service_events": list(getattr(controller, "events", []) or []),
            }
            failure_path = Path(args.runs_dir or "runs") / f"repair_result_{plan.plan_id}.json"
            failure_path.write_text(json.dumps(failure, indent=2, sort_keys=True))
            print(json.dumps(failure, indent=2))
            print(f"repair result -> {failure_path}")
            raise SystemExit(2)
    else:
        result = repair.apply_plan_with_live_status(
            client, cfg, plan, judge_mode=args.judge,
            judge_model=(cfg.judge_model if args.judge != "off" else None),
            rankings_dir=rankings_dir,
            live_ui=args.live_ui,
            ranking_scope=ranking_scope,
        )
    result_path = Path(args.runs_dir or "runs") / f"repair_result_{plan.plan_id}.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in result.items() if k != "actions"}, indent=2))
    print(f"repair result -> {result_path}")


def cmd_wizard(args, cfg):
    from . import wizard
    if args.samples is not None:
        cfg.samples = args.samples
    if getattr(args, "ctx", None):
        cfg.ctx_override = int(args.ctx)
    if getattr(args, "num_predict", None):
        cfg.num_predict_override = int(args.num_predict)
    if getattr(args, "think", None):
        cfg.think = args.think
    if getattr(args, "needle_max_ctx", None):
        cfg.needle_max_ctx = int(args.needle_max_ctx)
    if getattr(args, "judge_model", None):
        cfg.judge_model = args.judge_model
    client = _client(args, cfg)

    from .hardware import detect_gpus
    _wizard_inventory = detect_gpus()
    _wizard_identity_cache: dict = {}

    def _wizard_runtime_identities(selected_models):
        # Anvil Stage 3.6: bounded metadata only (same helper the run/
        # campaign paths use); lets the wizard's plan resolve the active
        # protocol / canonical benchmark runtime instead of deferring it.
        # The wizard rebuilds the plan on every keystroke -- reuse the
        # one GPU probe for the whole session, and memoize identities on
        # the selection (they only move when the selection does). A
        # failure here is non-fatal: the plan is built without identities.
        key = tuple(sorted(str(m) for m in selected_models))
        if key not in _wizard_identity_cache:
            try:
                _wizard_identity_cache[key] = _campaign_runtime_identities(
                    args, cfg, list(selected_models), client,
                    gpu_inventory=_wizard_inventory,
                )
            except Exception:
                _wizard_identity_cache[key] = {}
        return _wizard_identity_cache[key]

    plan, options = wizard.interactive_plan(
        client, cfg,
        initial_level=args.level,
        judge_mode=args.judge,
        initial_categories=_categories(args),
        initial_task_ids=_task_ids(args),
        runtime_identity_resolver=_wizard_runtime_identities,
        plan_kwargs={
            "include": args.include_regex,
            "exclude": args.exclude_regex,
            "skip_offload": args.skip_offload,
            "task_regex": args.task_regex,
            "family_base_only": args.family_base_only,
            "context_aliases_only": args.context_aliases_only,
            "context_only": args.context_only,
            "sample_mode": args.sample_mode,
        },
    )
    args.level = options["level"]
    args.categories = ",".join(options["categories"]) if options["categories"] else None
    args.tasks = ",".join(options["task_ids"]) if options["task_ids"] else None
    args.judge = options["judge_mode"]
    args.auto = True
    args.yes = True  # the wizard's explicit Accept action is the approval event
    args._selected_models = options["selected_models"]
    args._capability_profiles = options["capability_profiles"]
    args._accepted_plan = plan
    return cmd_run(args, cfg)


def cmd_diff(args, cfg):
    from . import compare
    a = Path(args.a)
    b = Path(args.b)
    out = Path(args.out) if args.out else b / "diff.md"
    text = compare.diff_runs(a, b, out, args.noise_band)
    print(text)
    print(f"\ndiff -> {out}")


def cmd_export_review(args, cfg):
    from . import compare
    runs = [Path(x) for x in args.runs]
    out = Path(args.out or "llm_modelbench_review_pack.zip")
    compare.export_review(runs, out)
    print(f"review pack -> {out}")


def cmd_repeat_report(args, cfg):
    from . import compare
    runs = [Path(x) for x in args.runs]
    out = Path(args.out) if args.out else None
    text = compare.repeatability_report(runs, out)
    print(text)
    if out:
        print(f"\nrepeatability report -> {out}")


def cmd_sensitivity_plan(args, cfg):
    from . import sensitivity
    print(sensitivity.plan_commands(
        run_prefix=args.run_prefix,
        include_regex=args.include_regex,
        tasks=args.tasks,
        level=args.level,
        ctx_values=args.ctx_values,
        num_predict_values=args.num_predict_values,
        judge=args.judge,
        fingerprint=args.fingerprint,
        needle_max_ctx=args.needle_max_ctx,
    ))


def cmd_sensitivity_report(args, cfg):
    from . import sensitivity
    text = sensitivity.report(args.runs)
    if args.out:
        Path(args.out).write_text(text)
        print(f"sensitivity report -> {args.out}")
    else:
        print(text)

def cmd_coverage(args, cfg):
    from . import coverage
    from .tasks import TASKS
    ledger_path = Path(args.ledger); ledger = coverage.load_ledger(ledger_path)
    if args.coverage_cmd == "update":
        run = Path(args.run_dir)
        rows = coverage.load_rows(run) if hasattr(coverage, "load_rows") else [json.loads(x) for x in (run / "raw_results.jsonl").read_text().splitlines() if x]
        identities = json.loads((run / "model_identities.json").read_text()) if (run / "model_identities.json").exists() else {}
        meta = json.loads((run / "summary_meta.json").read_text()) if (run / "summary_meta.json").exists() else {}
        coverage.update_ledger_from_run(ledger, raw_rows=rows, identities=identities, tasks=TASKS, benchmark_version=str(meta.get("benchmark_version") or "unknown"), out_dir=str(run), timestamp=str(meta.get("created_at") or ""))
        coverage.save_ledger(ledger, ledger_path); print(f"coverage ledger -> {ledger_path}")
    else:
        print(json.dumps(ledger, indent=2))

def cmd_rankings(args, cfg):
    if getattr(args, "adopt_campaign", None):
        from . import campaign
        paths = campaign.resolve_paths(args.adopt_campaign)
        if not paths.manifest.exists():
            raise SystemExit(f"unknown campaign {args.adopt_campaign!r}")
        preview = campaign.adopt_campaign(paths, rankings_dir=Path(args.out or "rankings"), dry_run=True)
        print(json.dumps(preview, indent=2, sort_keys=True))
        if args.dry_run:
            return
        required = f"ADOPT {args.adopt_campaign}"
        if not sys.stdin.isatty():
            raise SystemExit(f"canonical adoption requires typed terminal confirmation: {required}")
        if input(f"Type {required} to publish canonical rankings: ").strip() != required:
            raise SystemExit("canonical adoption cancelled")
        result = campaign.adopt_campaign(paths, rankings_dir=Path(args.out or "rankings"), dry_run=False)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    runs_dir = Path(args.runs_dir or "runs")
    rankings_dir = Path(args.out or "rankings")
    from . import ranking_controls

    changed = False
    if args.exclude_model:
        ranking_controls.set_model_excluded(rankings_dir, args.exclude_model, True, reason=args.reason)
        print(f"model excluded from rankings view -> {args.exclude_model}")
        changed = True
    if args.include_model:
        ranking_controls.set_model_excluded(rankings_dir, args.include_model, False, reason=args.reason)
        print(f"model included in rankings view -> {args.include_model}")
        changed = True
    if args.exclude_run:
        ranking_controls.set_run_excluded(rankings_dir, args.exclude_run, True, reason=args.reason)
        print(f"run excluded from rankings view -> {args.exclude_run}")
        changed = True
    if args.include_run:
        ranking_controls.set_run_excluded(rankings_dir, args.include_run, False, reason=args.reason)
        print(f"run included in rankings view -> {args.include_run}")
        changed = True
    if args.archive_run:
        ranking_controls.set_run_archived(rankings_dir, args.archive_run, True, reason=args.reason)
        print(f"run archived from normal rankings view -> {args.archive_run}")
        changed = True
    if args.unarchive_run:
        ranking_controls.set_run_archived(rankings_dir, args.unarchive_run, False, reason=args.reason)
        print(f"run unarchived for rankings view -> {args.unarchive_run}")
        changed = True
    if args.list_excluded:
        data = ranking_controls.load_exclusions(rankings_dir)
        print(json.dumps(data, indent=2, sort_keys=True))
        if not args.rescan and not changed and not args.watch:
            return

    if not args.watch:
        _update_rankings(
            runs_dir, rankings_dir, quiet=False, force=bool(args.rescan or changed),
            include_separate=bool(args.include_separate),
        )
        return
    print(f"watching {runs_dir} for ranking updates every {args.interval}s; Ctrl+C to stop")
    try:
        while True:
            _update_rankings(
                runs_dir, rankings_dir, quiet=False, force=bool(args.rescan or changed),
                include_separate=bool(args.include_separate),
            )
            args.rescan = False
            changed = False
            time.sleep(max(1.0, float(args.interval)))
    except KeyboardInterrupt:
        print("rankings watch stopped")


def _update_rankings(runs_dir: Path, rankings_dir: Path, quiet: bool, force: bool = False, *, include_separate: bool = False, only_run_ids=None) -> None:
    """Best-effort: called automatically after every completed run, and
    available standalone via `llmb rankings`. Never allowed to raise past
    this point -- a bug here must never fail an actual benchmark run."""
    try:
        from . import rankings
        template_path = Path(__file__).parent / "rankings_template.html"
        template = template_path.read_text() if template_path.exists() else None
        result = rankings.write_rankings(
            runs_dir, rankings_dir, html_template=template, force_rescan=force,
            include_separate=include_separate, only_run_ids=only_run_ids,
        )
        if not quiet:
            print(f"rankings updated: {result['models']} models, {result['raw_rows_total']} rows in the database")
            print(f"  raw     -> {result['raw_path']}")
            print(f"  summary -> {result['summary_path']}")
            print(f"  html    -> {result['html_path']}")
            if result.get("v3_html_path"):
                print(f"  v3      -> {result['v3_html_path']}")
            if result.get("v31_site_path"):
                print(f"  v3.1    -> {result['v31_site_path']}")
            if result.get("include_separate"):
                print("  scope   -> separate/diagnostic")
            exclusions = result.get("exclusions") or {}
            if any(exclusions.values()):
                print(f"  hidden  -> runs={exclusions.get('excluded_runs', 0) + exclusions.get('archived_runs', 0)} models={exclusions.get('excluded_models', 0)}")
    except Exception as exc:
        print(f"(rankings update skipped: {exc})")


def cmd_gaps(args, cfg):
    from .coverage import load_ledger
    from .gap_planner import gap_report
    from .tasks import TASKS
    client = _client(args, cfg); data = gap_report(client, load_ledger(Path(args.ledger)), TASKS, classify_model, families_for)
    print(json.dumps(data, indent=2) if args.json else "\n".join(f"{m}: {', '.join(c)}" for m,c in data.items()))

def cfg_weights_for(run: Path) -> dict:
    meta_path = run / "summary_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if isinstance(meta.get("category_weights"), dict):
            return meta["category_weights"]
    return {}

def _ledger_run_dirs(ledger: dict) -> set:
    return {entry.get("out_dir") for ledger_entry in ledger.values()
            for entry in ledger_entry.get("categories", {}).values() if entry.get("out_dir")}


def _run_row_aggregation_verdicts(run: Path) -> tuple[list, bool]:
    """For one ledger-referenced run dir, classify every scored row against
    today's canonical aggregation policy.

    Returns ``(row_verdicts, override_active)`` where ``row_verdicts`` is a list
    of ``(row, digest, AggregationPolicyVerdict)`` -- one per raw row that
    resolves to a digest -- and ``override_active`` is True when this run's own
    scoring used a weights override (recorded in ``summary_meta.json``).

    This is the single per-run policy-resolution path shared by the dossier's
    quality composite and its advisory surface; it reuses
    :func:`verify_recorded_aggregation_policy` (the one aggregation-policy
    authority, also used by canonical ranking) and does not rebuild any policy.
    """
    from .benchmark_policy import verify_recorded_aggregation_policy
    from .tasks import TASKS

    tasks_by_id = {task.id: task for task in TASKS}
    raw_path, identities_path = run / "raw_results.jsonl", run / "model_identities.json"
    if not raw_path.exists() or not identities_path.exists():
        return [], False
    identities = json.loads(identities_path.read_text())
    filters = json.loads((run / "filters.json").read_text()) if (run / "filters.json").exists() else {}
    bindings = json.loads((run / "benchmark_bindings.json").read_text()) if (run / "benchmark_bindings.json").exists() else {}
    run_meta = json.loads((run / "summary_meta.json").read_text()) if (run / "summary_meta.json").exists() else {}
    # per-run scoring override: report.py:_metadata records it here, and
    # cfg_weights_for() reads category_weights from the same file
    override_active = bool(run_meta.get("weight_override") or run_meta.get("category_weights"))
    protocols_by_key: dict = {}
    for entry in (bindings.get("bindings") or {}).values():
        key = ((entry or {}).get("binding") or {}).get("binding_key")
        protocol = (entry or {}).get("protocol")
        if key and isinstance(protocol, dict):
            protocols_by_key[str(key)] = protocol
    for entry in (bindings.get("resume_divergent_bindings") or []):
        key = ((entry or {}).get("binding") or {}).get("binding_key")
        protocol = (entry or {}).get("protocol")
        if key and isinstance(protocol, dict):
            protocols_by_key.setdefault(str(key), protocol)
    row_verdicts = []
    for line in raw_path.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        digest = (identities.get(row.get("model")) or {}).get("digest")
        if not digest:
            continue
        binding_key = row.get("benchmark_binding_key")
        protocol = protocols_by_key.get(str(binding_key)) if binding_key else None
        verdict = verify_recorded_aggregation_policy(
            recorded_hash=str((protocol or {}).get("aggregation_policy_hash") or ""),
            task_ids=list((protocol or {}).get("task_ids") or []),
            requested_samples=filters.get("requested_samples"),
            sample_mode=filters.get("sample_mode"),
            judge_mode=filters.get("judge_mode"),
            tasks_by_id=tasks_by_id,
        )
        row_verdicts.append((row, digest, verdict))
    return row_verdicts, override_active


def _quality_by_digest_from_ledger(ledger: dict) -> dict:
    """Load each ledger-referenced run and map aggregate category quality by
    digest.

    Anvil Stage 3.3C owner resolution: a canonical cross-run dossier composite
    must not combine evidence with provably incompatible aggregation semantics.
    Rows whose run's recorded aggregation policy has *drifted* from today's
    canonical policy (``policy_drift``) are dropped before ``aggregate()`` --
    exactly the canonical-ranking rule, one level up. ``unverified_legacy`` /
    ``unverified_incomplete`` rows are kept (no policy is retrospectively
    attributed to them); the per-run report never excludes anything.
    """
    from .aggregate import aggregate
    from .benchmark_policy import AGG_VERDICT_POLICY_DRIFT
    from .tasks import TASKS
    difficulty = {task.id: task.difficulty for task in TASKS}
    quality_by_digest = {}
    for out_dir in _ledger_run_dirs(ledger):
        run = Path(out_dir)
        identities_path = run / "model_identities.json"
        if not (run / "raw_results.jsonl").exists() or not identities_path.exists():
            continue
        row_verdicts, _ = _run_row_aggregation_verdicts(run)
        identities = json.loads(identities_path.read_text())
        rows = [
            row for row, _digest, verdict in row_verdicts
            if verdict.verdict != AGG_VERDICT_POLICY_DRIFT
        ]
        if not rows:
            continue
        _, per_cat = aggregate(rows, cfg_weights_for(run), difficulty)
        for category, ranked in per_cat.items():
            for model_name, quality in ranked:
                digest = (identities.get(model_name) or {}).get("digest")
                if digest:
                    quality_by_digest.setdefault(digest, {})[category] = quality
    return quality_by_digest

def _aggregation_policy_by_digest_from_ledger(ledger: dict) -> dict:
    """Per digest, count how each contributing ledger run's scored rows
    compare against today's canonical aggregation policy.

    Advisory surface for ``cmd_dossier``. Anvil Stage 3.3C owner resolution:
    ``policy_drift`` rows ARE excluded from the canonical cross-run composite
    (see :func:`_quality_by_digest_from_ledger`); this surface reports how many
    were excluded per digest and keeps them fully visible in
    ``verdict_counts`` / ``drift_reasons`` so the exclusion is never silent.
    ``unverified_legacy`` / ``unverified_incomplete`` rows are not excluded and
    keep their existing treatment. A run whose own scoring used a weights
    override (recorded in its ``summary_meta.json``) sets ``override_runs`` so a
    ``verified`` count is not read as a canonical endorsement; ``cmd_dossier``
    adds ``dossier_weights_overridden`` for its own ``--weights``.
    """
    from .benchmark_policy import AGG_VERDICT_POLICY_DRIFT

    by_digest: dict = {}
    for out_dir in _ledger_run_dirs(ledger):
        run = Path(out_dir)
        row_verdicts, override_active = _run_row_aggregation_verdicts(run)
        for _row, digest, verdict in row_verdicts:
            slot = by_digest.setdefault(
                digest,
                {"verdict_counts": {}, "drift_reasons": set(), "override_runs": False,
                 "excluded_from_canonical_composite": 0},
            )
            slot["verdict_counts"][verdict.verdict] = slot["verdict_counts"].get(verdict.verdict, 0) + 1
            slot["total_rows"] = slot.get("total_rows", 0) + 1
            if verdict.verdict == AGG_VERDICT_POLICY_DRIFT:
                slot["excluded_from_canonical_composite"] += 1
            if verdict.reason:
                slot["drift_reasons"].add(verdict.reason)
            if override_active:
                slot["override_runs"] = True
    return {
        digest: {
            "verdict_counts": dict(sorted(slot["verdict_counts"].items())),
            "drift_reasons": sorted(slot["drift_reasons"]),
            "override_runs": slot["override_runs"],
            "excluded_from_canonical_composite": slot["excluded_from_canonical_composite"],
            "all_rows_excluded_as_drift": (
                slot["excluded_from_canonical_composite"] > 0
                and slot["excluded_from_canonical_composite"] == slot.get("total_rows", 0)
            ),
            "note": (
                "advisory; policy_drift rows are excluded from the canonical "
                "dossier composite but remain counted here"
            ),
        }
        for digest, slot in by_digest.items()
    }

def cmd_dossier(args, cfg):
    from .coverage import load_ledger
    from .dossier import DEFAULT_CATEGORY_WEIGHTS, composite_score, validate_weights
    from .weights_override import parse_weight_overrides
    from .tasks import TASKS
    dossier_weights_overridden = bool(args.weights)
    weights = parse_weight_overrides(args.weights, DEFAULT_CATEGORY_WEIGHTS) if args.weights else DEFAULT_CATEGORY_WEIGHTS
    validate_weights(weights); ledger = load_ledger(Path(args.ledger)); out = {}; quality_by_digest = _quality_by_digest_from_ledger(ledger)
    aggregation_policy_by_digest = _aggregation_policy_by_digest_from_ledger(ledger)
    for digest, entry in ledger.items():
        policy = dict(aggregation_policy_by_digest.get(
            digest,
            {"verdict_counts": {}, "drift_reasons": [], "override_runs": False,
             "excluded_from_canonical_composite": 0, "all_rows_excluded_as_drift": False},
        ))
        # the dossier's own --weights override also makes this composite
        # non-canonical, independent of any per-run override
        policy["dossier_weights_overridden"] = dossier_weights_overridden
        policy["canonical_composite"] = not dossier_weights_overridden and not policy.get("override_runs", False)
        composite = composite_score(digest, ledger, quality_by_digest.get(digest, {}), weights, TASKS)
        # Stage 3.3C: policy_drift rows were dropped upstream. When *every*
        # contributing row for this digest was excluded as drift, the composite
        # is honestly unavailable rather than a live-policy number over
        # incompatible rows -- flag the reason so it is not read as mere missing
        # coverage. (A composite that is None only because of stale/absent
        # ledger coverage keeps the existing, unflagged meaning.)
        if policy.get("all_rows_excluded_as_drift") and composite.get("composite") is None:
            policy["canonical_composite_unavailable_reason"] = (
                "all comparable rows excluded as policy_drift"
            )
        out[digest] = composite | {
            "names_seen": entry.get("names_seen", []),
            "aggregation_policy": policy,
        }
    text=json.dumps(out, indent=2)
    if args.out: Path(args.out).write_text(text)
    if args.json or not args.out: print(text)


# Shared run/plan arguments. Keep the wizard/doctor simple and the core CLI scriptable.
def _add_run_filters(
    r, *, include_model_selection: bool = True, include_auto: bool = True,
    auto_default: bool = False,
):
    r.add_argument("--runtime-profile", help="saved runtime profile; explicit selection takes precedence")
    r.add_argument("--level", choices=["smoke", "short", "full"], default="smoke")
    r.add_argument("--categories", help="comma-separated category filter")
    r.add_argument("--tasks", help="comma-separated task IDs to run, e.g. py_anagram,json_extract,needle")
    r.add_argument("--task-regex", help="regex over task id/category/scorer, e.g. needle|context")
    r.add_argument("--family-base-only", action="store_true",
                   help="skip obvious context aliases such as 64k/128k/ctx/exp variants")
    r.add_argument("--context-aliases-only", action="store_true",
                   help="run only obvious context aliases such as 64k/128k/ctx/exp variants")
    r.add_argument("--context-only", action="store_true",
                   help="run only long-context/needle tasks; combine with --context-aliases-only for alias validation")
    r.add_argument(
        "--allow-host-code-execution", action="store_true",
        help=("explicitly permit deterministic scorers to execute model-generated Python/JavaScript "
              "on this host; use only inside a disposable container or VM"),
    )
    if include_model_selection:
        selection = r.add_mutually_exclusive_group()
        selection.add_argument("--models", help="exact installed model names separated by semicolons")
        selection.add_argument("--all", dest="all_models", action="store_true",
                               help="explicitly select every model returned by Ollama (also the default when no selector is given)")
        selection.add_argument("--select", action="store_true",
                               help="interactive MODEL selector only; test scope still comes from --level/--categories/--tasks")
    if include_auto:
        auto_group = r.add_mutually_exclusive_group()
        auto_group.add_argument(
            "--auto", dest="auto", action="store_true",
            help="run small functional capability probes before routing tasks",
        )
        auto_group.add_argument(
            "--no-auto-probe", dest="auto", action="store_false",
            help="route from metadata/operator profiles only; skips pre-run functional probes",
        )
        r.set_defaults(auto=bool(auto_default))
    r.add_argument("--include-regex")
    r.add_argument("--exclude-regex")
    r.add_argument("--skip-offload", action="store_true", help="skip models that exceed the VRAM budget")
    r.add_argument("--ctx", type=int, help="override Ollama num_ctx for all chat calls in this run")
    r.add_argument("--num-predict", type=int, help="override Ollama num_predict for all normal task generations")
    r.add_argument("--think", choices=["auto", "on", "off"], default=None,
                   help="control Ollama thinking where supported; auto leaves server/model default")
    r.add_argument("--needle-max-ctx", type=int,
                   help="operator safety cap for needle probe num_ctx; larger probes are skipped and coverage drops")
    r.add_argument("--judge-model", help="local Ollama model used for subjective judging, default from config/env")
    r.add_argument("--samples", type=int, help="requested runs per sampled task; smart mode applies this only to judged tasks")
    r.add_argument("--sample-mode", choices=["smart", "all"], default="smart",
                   help="smart=sample only subjective/judged tasks; all=old behavior, sample every task")
    r.add_argument("--mock", action="store_true", help="run fully offline against a deterministic stub")
    r.add_argument("--plan-json", help="write the computed plan JSON to this path")
    return r


def build_parser():
    p = argparse.ArgumentParser(prog="llm-modelbench",
                                description="Hardware-adaptive benchmark suite for local Ollama models.")
    p.add_argument("--version", action="version", version=f"llm-modelbench {__version__}")
    p.add_argument("--config", help="path to a JSON or YAML config file")
    p.add_argument("--runtime-profiles-file", help=argparse.SUPPRESS)
    p.add_argument("--selftest", action="store_true", help="run offline scorer tests and exit")
    sub = p.add_subparsers(dest="cmd")

    inv = sub.add_parser("inventory", help="list local models")
    inv.add_argument("--json", action="store_true")
    inv.add_argument("--mock", action="store_true", help="use offline stub model list")
    inv.add_argument("--auto", action="store_true", help="also run functional capability probes")
    inv.add_argument("--runtime-profile", help="saved runtime profile; explicit selection takes precedence")

    ce = sub.add_parser(
        "capability-evidence",
        help="read-only classification of the fleet's stored capability evidence (Anvil Stage 2.7A)",
    )
    ce.add_argument("--json", action="store_true", help="write the full machine-readable report to stdout")
    ce.add_argument("--mock", action="store_true", help="use offline stub model list")
    ce.add_argument("--runtime-profile", help="saved runtime profile; explicit selection takes precedence")
    ce.add_argument("--runs-dir", default="runs", help="root directory to scan for capability_report.json (default: runs)")
    ce.add_argument("--campaigns-dir", default="campaigns", help="root directory to scan for campaign evidence (default: campaigns)")
    ce.add_argument("--reprobe-required-only", action="store_true", help="in text mode, list only cells that need a reprobe")

    rp = sub.add_parser(
        "reprobe-plan",
        help="deterministic, read-only capability reprobe plan from Stage 2.7A evidence classification (Anvil Stage 2.7B)",
    )
    rp.add_argument("--json", action="store_true", help="write the full machine-readable plan to stdout")
    rp.add_argument("--mock", action="store_true", help="use offline stub model list")
    rp.add_argument("--runtime-profile", help="saved runtime profile; explicit selection takes precedence")
    rp.add_argument("--runs-dir", default="runs", help="root directory to scan for capability_report.json (default: runs)")
    rp.add_argument("--campaigns-dir", default="campaigns", help="root directory to scan for campaign evidence (default: campaigns)")
    rp.add_argument("--model", help="filter to one model alias")
    rp.add_argument("--capability", choices=list(FAMILY_ORDER), help="filter to one capability family")
    rp.add_argument("--backend", help="filter to one current backend")
    rp.add_argument("--reason", help="filter to one Stage 2.7A classification bucket (e.g. model_identity_changed)")
    rp.add_argument("--only-required", action="store_true", help="show only cells that need a reprobe")

    rx = sub.add_parser(
        "reprobe-execute",
        help="execute a Stage 2.7B reprobe plan's REPROBE actions and append evidence (Anvil Stage 2.7C)",
    )
    rx.add_argument("--json", action="store_true", help="write the full machine-readable execution report to stdout")
    rx.add_argument("--mock", action="store_true", help="use offline stub model list")
    rx.add_argument("--runtime-profile", help="saved runtime profile; explicit selection takes precedence")
    rx.add_argument("--runs-dir", default="runs", help="root directory to scan for capability_report.json (default: runs)")
    rx.add_argument("--campaigns-dir", default="campaigns", help="root directory to scan for campaign evidence (default: campaigns)")
    rx.add_argument("--ledger-path", help="EvidenceLedger JSONL file (default: <runs-dir>/capability_evidence_ledger.jsonl)")
    rx.add_argument("--model", help="filter to one model alias")
    rx.add_argument("--capability", choices=list(FAMILY_ORDER), help="filter to one capability family")
    rx.add_argument("--backend", help="filter to one current backend")
    rx.add_argument("--reason", help="filter to one Stage 2.7A classification bucket (e.g. model_identity_changed)")
    rx.add_argument("--apply", action="store_true", help="actually run probes and append evidence; default is dry-run (plan only)")

    fit = sub.add_parser("runtime-fit", help="read-only conservative model-to-runtime GPU capacity assessment")
    fit.add_argument("--model", required=True, help="exact model name already known to the selected runtime")
    fit.add_argument("--runtime-profile", help="saved runtime profile; explicit selection takes precedence")
    fit.add_argument("--context", type=int, help="requested context; never silently clamped")
    fit.add_argument("--reserve-mib", type=float, default=512.0, help="per-device safety reserve in MiB")
    fit.add_argument("--strategy", choices=["layer_split", "tensor_split"], help="explicit multi-GPU strategy declaration")
    fit.add_argument("--allocation-weights", help="comma-separated GPU-UUID=weight layer allocations; never positional")
    fit.add_argument("--allow-cpu-spill", action="store_true",
                     help="advisory only: mark spill operator-permitted for this estimate. Host RAM capacity is "
                          "not checked here; actual RAM-spill feasibility is decided by the benchmark execution preflight")
    fit.add_argument("--json", action="store_true", help="write deterministic JSON to stdout")
    fit.add_argument("--out", help="optional JSON output path")
    fit.add_argument("--mock", action="store_true", help="use the offline deterministic model client")

    doc = sub.add_parser("doctor", help="preflight environment, import path, Ollama, GPU, disk")
    doc.add_argument("--json", action="store_true")

    runtime = sub.add_parser("runtime", help="discover and manage external local runtime profiles")
    runtime_sub = runtime.add_subparsers(dest="runtime_cmd", required=True)
    runtime_sub.add_parser("discover", help="bounded read-only local runtime discovery")
    runtime_sub.add_parser("list", help="list saved runtime profiles")
    runtime_show = runtime_sub.add_parser("show", help="show one saved runtime profile")
    runtime_show.add_argument("name")
    runtime_save = runtime_sub.add_parser("save", help="save one external runtime profile")
    runtime_save.add_argument("--name", required=True)
    runtime_save.add_argument("--backend", required=True, choices=["ollama", "llama_cpp"])
    runtime_save.add_argument("--endpoint", required=True)
    runtime_save.add_argument("--gpu-uuid", action="append")
    runtime_save.add_argument("--description")
    runtime_save.add_argument("--provenance", choices=["configured", "discovered", "legacy-default"], default="configured")
    runtime_save.add_argument("--set-default", action="store_true")
    runtime_save.add_argument("--replace", action="store_true")
    runtime_save.add_argument("--yes", action="store_true")
    runtime_delete = runtime_sub.add_parser("delete", help="delete only one saved runtime profile")
    runtime_delete.add_argument("name")
    runtime_delete.add_argument("--yes", action="store_true")
    runtime_select = runtime_sub.add_parser("select", help="select a healthy discovered runtime")
    runtime_select.add_argument("--runtime-profile")
    runtime_select.add_argument("--save-name")
    runtime_select.add_argument("--set-default", action="store_true")
    runtime_select.add_argument("--replace", action="store_true")
    runtime_select.add_argument("--yes", action="store_true")

    pl = sub.add_parser("plan", help="show active models, skipped models, tasks, samples, and rough ETA without running")
    _add_run_filters(pl)
    pl.add_argument("--judge", choices=["single", "panel", "off"], default="off",
                    help="subjective scoring mode used for sample planning; default: off")
    pl.add_argument("--json", action="store_true")

    camp = sub.add_parser(
        "campaign",
        help="manage isolated campaign workspaces",
        description=(
            "Manage isolated campaign workspaces. Config execution is strict and "
            "records an immutable plan signature; supersession appends validated "
            "evidence and never rewrites primary rows."
        ),
    )
    camp_sub = camp.add_subparsers(dest="campaign_cmd", required=True)
    camp_init = camp_sub.add_parser("init", help="write a declarative campaign JSON template")
    camp_init.add_argument("path", nargs="?", default="campaign.json")
    camp_execute = camp_sub.add_parser(
        "execute",
        help="execute strict campaign config with immutable plan signature",
        description=(
            "Validate a campaign JSON file, execute the normal campaign lifecycle, "
            "and persist the immutable plan signature. Existing campaigns must "
            "have the same signature; interrupted campaigns require campaign resume."
        ),
    )
    camp_execute.add_argument("--config", dest="campaign_config", required=True, help="strict campaign JSON created by campaign init")
    camp_execute.add_argument("--mock", action="store_true", help="use the deterministic offline mock backend")
    camp_supersede = camp_sub.add_parser(
        "supersede",
        help="append immutable corrected-evidence supersession",
        description=(
            "Append one schema-versioned source-to-replacement evidence edge. "
            "The source hash must identify primary evidence; forks, cycles, "
            "unsupported schemas, and hash/provenance contradictions fail closed."
        ),
    )
    camp_supersede.add_argument("--campaign-id", required=True)
    camp_supersede.add_argument("--source-campaign-id")
    camp_supersede.add_argument("--source-run-id", default="primary")
    camp_supersede.add_argument("--source-row", help="JSON file used only to derive/confirm the immutable source row hash")
    camp_supersede.add_argument("--source-row-hash", help="immutable source row hash from campaign primary evidence")
    camp_supersede.add_argument("--replacement-campaign-id")
    camp_supersede.add_argument("--replacement-run-id", required=True)
    camp_supersede.add_argument("--replacement-row", required=True, help="JSON file containing the corrected row")
    camp_supersede.add_argument("--replacement-row-hash", help="expected corrected row hash")
    camp_supersede.add_argument("--reason", required=True)
    camp_supersede.add_argument("--operator", default="operator")
    camp_supersede.add_argument("--dry-run", action="store_true", help="validate and preview without appending")
    camp_status = camp_sub.add_parser("status", help="show campaign lifecycle state")
    camp_status.add_argument("campaign_id")
    camp_resume = camp_sub.add_parser("resume", help="resume the exact recorded interrupted campaign phase")
    camp_resume.add_argument("campaign_id")
    camp_resume.add_argument("--runtime-profile", help="saved runtime profile; explicit selection takes precedence")
    camp_package = camp_sub.add_parser("package", help="write one campaign review package")
    camp_package.add_argument("campaign_id")
    camp_clean = camp_sub.add_parser("clean", help="preview or apply conservative retained-evidence cleanup")
    camp_clean.add_argument("campaign_id", nargs="?")
    camp_clean.add_argument("--all", action="store_true", help="process all eligible campaigns and report unsafe skips")
    camp_clean_mode = camp_clean.add_mutually_exclusive_group()
    camp_clean_mode.add_argument("--apply", action="store_true", help="remove only listed disposable campaign dumps")
    camp_clean_mode.add_argument("--dry-run", action="store_false", dest="apply", help="preview only (default)")
    camp_migrate = camp_sub.add_parser("migrate-legacy", help="copy a legacy run into a campaign")
    camp_migrate.add_argument("--run-id", required=True)
    camp_migrate.add_argument("--campaign-id", required=True)
    camp_migrate.add_argument("--runs-dir", default="runs")
    camp_migrate_mode = camp_migrate.add_mutually_exclusive_group()
    camp_migrate_mode.add_argument("--apply", action="store_true")
    camp_migrate_mode.add_argument("--dry-run", action="store_false", dest="apply", help="preview only (default)")
    camp_plan = camp_sub.add_parser(
        "plan",
        help="create an isolated campaign plan",
        description=(
            "Create or inspect the immutable pre-generation campaign plan. "
            "This command does not run fingerprint probes; --no-fingerprint is "
            "available only on campaign run."
        ),
    )
    camp_plan.add_argument("--campaign-id", required=True)
    _add_run_filters(camp_plan)
    camp_plan.add_argument("--judge", choices=["off"], default="off", help="campaign primary generation is planned with judge mode off")
    camp_run = camp_sub.add_parser("run", help="run a primary benchmark inside a campaign")
    camp_run.add_argument("--campaign-id", required=True)
    _add_run_filters(camp_run, auto_default=True)
    camp_run.add_argument("--judge", choices=["single", "panel", "off"], default="off")
    camp_run.add_argument("--dump-subjective", action="store_true", default=True)
    camp_run.add_argument("--no-dump", dest="dump_subjective", action="store_false")
    camp_run.add_argument("--dump-raw", action="store_true", default=True)
    camp_run.add_argument("--no-dump-raw", dest="dump_raw", action="store_false")
    camp_run.add_argument("--fingerprint", action="store_true", default=True)
    camp_run.add_argument("--no-fingerprint", dest="fingerprint", action="store_false")
    camp_run.add_argument("--resume", action="store_true", default=True)
    camp_run.add_argument("--no-resume", dest="resume", action="store_false")
    camp_run.add_argument("--yes", action="store_true")
    camp_run.add_argument("--status-interval", type=float, default=5.0)
    camp_run.add_argument("--live-ui", choices=["off", "compact", "full", "graph", "log"], default="compact")
    camp_run.add_argument("--strict-harness", action="store_true")
    camp_run.add_argument("--allow-ram-spill", action="store_true",
                          help="permit physical-RAM fallback only after GPU capacity is proven insufficient; "
                               "conservative host-RAM preflight still gates it. No other authority is implied. Default: off")
    camp_run.add_argument("--unattended-safe", action="store_true", help="write terminal readiness and review package without host mutation")
    camp_run.add_argument("--unattended", action="store_true",
                          help="use the unattended decision policy: complete without interactive stdin where a decision "
                               "is explicitly safe to make automatically (e.g. a decisive backend recommendation), "
                               "otherwise fail closed with a typed reason. Independent of --yes/--unattended-safe/"
                               "--auto-confirm, which keep their existing meanings and are not implied by this flag.")

    r = sub.add_parser("run", help="run the benchmark")
    # Actual scored runs probe capability lanes by default. Planning remains
    # metadata-only unless --auto is explicit, so a read-only plan stays cheap.
    _add_run_filters(r, auto_default=True)
    r.add_argument("--judge", choices=["single", "panel", "off"], default="off",
                   help="subjective scoring: single judge, persona panel, or off (dump only). Default: off")
    r.add_argument("--dump-subjective", action="store_true", default=True,
                   help="also save subjective outputs for human grading")
    r.add_argument("--no-dump", dest="dump_subjective", action="store_false")
    r.add_argument("--dump-raw", action="store_true", default=True,
                   help="save deterministic raw outputs under raw/<task>/<model>.txt (default on)")
    r.add_argument("--no-dump-raw", dest="dump_raw", action="store_false")
    r.add_argument("--fingerprint", action="store_true", default=True,
                   help="run clone fingerprint probes when plan size is sufficient")
    r.add_argument("--no-fingerprint", dest="fingerprint", action="store_false")
    r.add_argument("--run-id", help="stable run directory name for resume")
    r.add_argument("--out", help="base output directory (default: runs)")
    r.add_argument("--resume", action="store_true", default=True)
    r.add_argument("--no-resume", dest="resume", action="store_false")
    r.add_argument("--yes", action="store_true", help="accept the printed run plan without prompting")
    r.add_argument("--status-interval", type=float, default=5.0,
                   help="seconds between status updates when supported (status.json always updates on task events)")
    r.add_argument("--live-ui", choices=["off", "compact", "full", "graph", "log"], default="compact",
                   help="inline dashboard: compact/full/graph/log/off; d dashboard, l log, q stop after current task")
    r.add_argument("--rankings-out", help="rankings directory refreshed after the run; default: rankings")
    r.add_argument("--no-ranking-update", action="store_true", help="write evidence but skip automatic rankings refresh")
    r.add_argument("--strict-harness", action="store_true",
                   help="exit nonzero if any selected task ends in a harness/resource/configuration error")
    r.add_argument("--separate-ranking", action="store_true", help="write evidence and generate an isolated rankings-separate/<run-id> report instead of touching canonical rankings")
    r.add_argument("--allow-ram-spill", action="store_true",
                   help="permit physical-RAM fallback only after GPU capacity is proven insufficient; "
                        "conservative host-RAM preflight still gates it. No other authority is implied. Default: off")
    r.add_argument("--unattended", action="store_true",
                   help="use the unattended decision policy: complete without interactive stdin where a decision "
                        "is explicitly safe to make automatically (e.g. a decisive backend recommendation), "
                        "otherwise fail closed with a typed reason. Independent of --yes, which keeps its "
                        "existing meaning and is not implied by this flag.")

    w = sub.add_parser("watch", help="live terminal dashboard for a run")
    w.add_argument("--run-id", help="which run to watch; if omitted, auto-detects "
                   "from --runs-dir (auto-picks if unambiguous, otherwise prompts)")
    w.add_argument("--out", help="run directory or base output directory (default: runs when --run-id is used)")
    w.add_argument("--runs-dir", default="runs", help="where to look for runs when --run-id/--out are omitted")
    w.add_argument("--layout", choices=["full", "compact", "bars", "failures", "hardware", "repair", "context", "interactive"], default="full")
    w.add_argument("--refresh", type=float, default=1.0)
    w.add_argument("--no-clear", action="store_true", help="append frames instead of redrawing the terminal")
    w.add_argument("--follow-queue", dest="follow_queue", action="store_true", default=None,
                   help="follow the whole queue: auto-advance to whatever run starts next once the "
                        "current one finishes, until nothing new appears, then print a summary and exit. "
                        "This is the default when no --run-id/--out/--once is given.")
    w.add_argument("--no-follow-queue", dest="follow_queue", action="store_false",
                   help="opt out of queue-following; watch whatever's auto-picked (or --run-id/--out) "
                        "once, the old single-run behavior")
    w.add_argument("--idle-grace", type=float, default=180.0,
                   help="with queue-following, seconds with no new run appearing before concluding "
                        "the queue is finished (default 180)")
    w.add_argument("--screen", choices=["auto", "alternate", "normal", "scroll"], default="auto",
                   help="rendering mode: auto/alternate keeps a single dashboard window; normal redraws current screen; scroll appends")
    w.add_argument("--once", action="store_true", help="render one dashboard frame and exit")
    w.add_argument("--exit-when-done", action="store_true",
                   help="exit when a repair campaign reaches complete/partial/failed")
    w.add_argument("--mock", action="store_true")

    rep = sub.add_parser("report", help="rebuild reports for a run")
    rep.add_argument("--run-id")
    rep.add_argument("--out", help="run directory (if not using --run-id)")
    rep.add_argument("--weights", help="report-time category overrides, e.g. coding_python=0.4,agentic_tool=0.3")
    rep.add_argument("--report-out", help="separate output directory for --weights; defaults to a sibling override copy")
    rep.add_argument("--mock", action="store_true")

    sim = sub.add_parser("simulate", help="offline VRAM and watcher simulations")
    sim.add_argument("--run-dir", help="legacy: finished run directory containing raw_results.jsonl")
    sim.add_argument("--simulate-vram", type=float, help="legacy: hypothetical VRAM budget in GB")
    sim.add_argument("--json", action="store_true")
    sim_sub = sim.add_subparsers(dest="simulate_cmd")
    sim_watch = sim_sub.add_parser(
        "repair-watch",
        help="replay deterministic repair status transitions without Ollama or GPU work",
    )
    sim_watch.add_argument("--scenario", choices=[
        "capability-repair", "needle-current", "kv-cascade",
        "interrupted-child", "failed-child",
    ], default="capability-repair")
    sim_watch.add_argument("--speed", type=float, default=1.0,
                           help="seconds between deterministic status transitions; 0 runs immediately")
    sim_watch.add_argument("--runs-dir", default="runs")
    sim_watch.add_argument("--run-id", help="stable fixture campaign directory name")
    sim_watch.add_argument("--write-only", action="store_true",
                           help="write fixture files without rendering; attach llmb-watch separately")
    sim_watch.add_argument("--cleanup", action="store_true",
                           help="remove fixture directories after replay")
    sim_watch.add_argument("--screen", choices=["auto", "normal", "scroll"], default="auto")
    sim_watch.add_argument("--json", action="store_true")

    cp = sub.add_parser("context-profile", help="run one controlled 64k-class needle telemetry profile")
    cp.add_argument("--model", required=True, help="exact installed Ollama model name")
    cp.add_argument("--runtime-profile", help="saved runtime profile; explicit selection takes precedence")
    cp.add_argument("--run-id")
    cp.add_argument("--runs-dir", default="runs")
    cp.add_argument("--rankings-out", help="rankings directory refreshed after the profile; default: rankings")
    cp.add_argument("--cards-out", default="model_cards")
    cp.add_argument("--target-ctx", type=int, default=64000)
    cp.add_argument("--gpu-vram-gb", type=float)
    cp.add_argument("--emergency-headroom-gb", type=float, default=0.25)
    cp.add_argument("--max-spill-gb", type=float, default=2.5)
    cp.add_argument("--min-tps", type=float, default=10.0)
    cp.add_argument("--critical-tps", type=float, default=3.0)
    cp.add_argument("--live-ui", choices=["off", "compact", "full", "graph", "log"], default="compact")
    cp.add_argument("--behavior-probe", dest="behavior_probe", action="store_true", default=True,
                    help="also run a synthetic 64k recall/structure/speed probe (default on)")
    cp.add_argument("--no-behavior-probe", dest="behavior_probe", action="store_false",
                    help="skip the synthetic behavior probe; telemetry validation will cover needle only")
    cp.add_argument("--yes", action="store_true")
    cp.add_argument("--mock", action="store_true")
    cp.add_argument("--no-ranking-update", action="store_true", help="write the diagnostic run but skip rankings/model-card refresh")
    cp.add_argument("--separate-ranking", action="store_true", help="generate an isolated rankings-separate/<run-id> report for this diagnostic profile")

    mc = sub.add_parser("model-cards", help="generate standalone operating cards from master rankings")
    mc.add_argument("--rankings-dir", default="rankings")
    mc.add_argument("--runs-dir", default="runs")
    mc.add_argument("--out", default="model_cards")

    fr = sub.add_parser("freeze", help="create a pre-release source/task/rankings regression snapshot")
    fr.add_argument("--repo-root", default=".")
    fr.add_argument("--runs-dir", default="runs")
    fr.add_argument("--rankings-dir", default="rankings")
    fr.add_argument("--out", required=True)
    fr.add_argument("--label", default="pre-rankings-v3")
    fr.add_argument("--no-rankings-copy", action="store_true")
    fr.add_argument("--verify", action="store_true",
                    help="verify an existing snapshot at --out without rebuilding it")

    srv = sub.add_parser("serve", help="serve read-only routing data from summary.json artifacts")
    srv.add_argument("--runs-dir", action="append", required=True, help="repeatable finished run directory")
    srv.add_argument("--port", type=int, default=8756)
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--allow-remote", action="store_true",
                     help="permit binding outside loopback; exposes model/routing metadata on the network")
    srv.add_argument("--allow-empty", action="store_true",
                     help="start even when no valid summary.json artifacts were loaded")

    pk = sub.add_parser("pack-subjective", help="collate subjective outputs for grading")
    pk.add_argument("--run-id")
    pk.add_argument("--out")
    pk.add_argument("--mock", action="store_true")

    gr = sub.add_parser("grade", help="blind human grading workflow for subjective outputs")
    gr.add_argument("--run-id")
    gr.add_argument("--out")
    gr.add_argument("--export-blind", action="store_true", help="write a blind grading pack without prompting")
    gr.add_argument("--mock", action="store_true")

    jd = sub.add_parser("judge-dumps", help="judge existing subjective dumps without rerunning tested models")
    target = jd.add_mutually_exclusive_group(required=True)
    target.add_argument("--run-id", help="one run under --runs-dir")
    target.add_argument("--out", help="one explicit run directory")
    target.add_argument("--everything", action="store_true",
                        help="scan every run under --runs-dir and process eligible subjective dumps sequentially")
    jd.add_argument("--runs-dir", default="runs", help="run root for --run-id or --everything")
    jd.add_argument("--rankings-out", help="rankings database to refresh after judging; default: rankings")
    jd.add_argument("--judge", choices=["single", "panel"], default="single")
    jd.add_argument("--judge-model", help="local Ollama judge model")
    jd.add_argument("--ctx", type=int, help="judge context override")
    jd.add_argument("--think", choices=["auto", "on", "off"], default=None)
    jd.add_argument("--dry-run", action="store_true", help="scan and print eligibility without calling the judge")
    jd.add_argument("--force", action="store_true", help="rejudge rows already judged by the same model/mode")
    jd.add_argument("--yes", action="store_true", help="approve the printed judge batch without an interactive prompt")
    jd.add_argument("--mock", action="store_true", help="offline deterministic judge for pipeline testing")
    jd.add_argument("--runtime-profile", help="saved runtime profile; explicit selection takes precedence")
    jd.add_argument("--no-ranking-update", action="store_true", help="write judgements but skip automatic rankings refresh")
    jd.add_argument("--separate-ranking", action="store_true", help="generate an isolated rankings-separate/<run-id> report after judging")

    rp = sub.add_parser("repair", help="scan incomplete run evidence and apply bounded targeted recovery")
    repair_target = rp.add_mutually_exclusive_group(required=True)
    repair_target.add_argument("--run-id", help="repair one run under --runs-dir")
    repair_target.add_argument("--run-prefix", help="repair every run whose ID starts with this prefix")
    repair_target.add_argument("--everything", action="store_true", help="scan every run under --runs-dir")
    rp.add_argument("--runs-dir", default="runs")
    rp.add_argument("--rankings-out", help="rankings directory refreshed after --apply; default: rankings")
    rp.add_argument("--plan-out", help="write the repair plan JSON to this path")
    repair_mode = rp.add_mutually_exclusive_group()
    repair_mode.add_argument("--apply", action="store_true", help="execute the printed bounded repair plan")
    repair_mode.add_argument("--dry-run", action="store_true", help="explicitly request planning only; this is also the default when --apply is omitted")
    rp.add_argument("--yes", action="store_true", help="approve the printed repair plan in non-interactive execution")
    rp.add_argument("--judge", choices=["off", "single", "panel"], default="off",
                    help="post-hoc judge eligible subjective dumps as part of repair")
    rp.add_argument("--judge-model", help="local Ollama judge model")
    rp.add_argument("--num-predict", type=int, default=4096,
                    help="bounded output budget for thinking-only/empty-output recovery")
    rp.add_argument("--no-transient-retry", action="store_true",
                    help="report HTTP 5xx/timeouts but do not schedule one retry")
    rp.add_argument("--no-missing-tasks", action="store_true",
                    help="repair only explicit failed rows, not absent/stale applicable tasks")
    rp.add_argument("--emergency-headroom-gb", type=float, default=0.25,
                    help="physical VRAM kept free during guarded needle planning")
    rp.add_argument("--max-spill-gb", type=float, default=2.0,
                    help="maximum estimated system-RAM spill permitted for guarded needle retries")
    rp.add_argument("--kv-type", choices=["current", "q8_0", "q4_0"], default="current",
                    help="required Ollama KV-cache type for needle repair; explicit values require server setup/restart")
    rp.add_argument("--gpu-vram-gb", type=float,
                    help="override detected physical GPU VRAM for offline planning or unusual drivers")
    rp.add_argument("--confirm-kv-server", action="store_true",
                    help="assert that the running Ollama service was restarted with --kv-type when process environment cannot be inspected")
    rp.add_argument("--kv-cascade", action="store_true",
                    help="current/default-KV first; use q8_0 then unresolved-only q4_0 only for remaining guarded needle work")
    rp.add_argument("--restart-ollama", action="store_true",
                    help="allow the explicit KV cascade to install a temporary systemd drop-in and restart Ollama")
    rp.add_argument("--ollama-service", default="auto",
                    help="deprecated display hint; the privileged broker always discovers the sole Ollama unit owning the configured port")
    rp.add_argument("--keep-final-kv", action="store_true",
                    help="do not restore the original Ollama service drop-in after the cascade")
    rp.add_argument("--reuse-sudo-credentials", action="store_true",
                    help="do not invalidate sudo's cached timestamp before each privileged phase")
    rp.add_argument("--auto-confirm", action="store_true",
                    help="fully unattended apply mode for --restart-ollama --kv-cascade: implies --yes, skips "
                         "typed DISCOVER/VERIFY/RESTART confirmations, and uses sudo -n only. Requires a scoped "
                         "NOPASSWD sudoers rule for the dedicated broker in docs/auto_confirm_sudoers.md. Does "
                         "not store or read a password. Off by default.")
    rp.add_argument("--force", action="store_true", help="allow a previously recorded repair action to be planned again")
    rp.add_argument("--live-ui", choices=["off", "compact", "full", "log"], default="compact",
                    help="inline repair-aware dashboard for child runs; detached llmb-watch remains supported")
    rp.add_argument("--mock", action="store_true", help="offline deterministic repair pipeline test")
    rp.add_argument("--runtime-profile", help="saved runtime profile; explicit selection takes precedence")
    rp.add_argument("--no-ranking-update", action="store_true", help="write repair evidence but skip automatic rankings refresh")
    rp.add_argument("--separate-ranking", action="store_true", help="generate an isolated rankings-separate/<plan-id> report for repair children")

    wz = sub.add_parser("wizard", help="interactive model + test-scope planner, capability probe, review, and run")
    _add_run_filters(wz, include_model_selection=False, include_auto=False)
    wz.add_argument("--judge", choices=["single", "panel", "off"], default="off")
    wz.add_argument("--dump-subjective", action="store_true", default=True)
    wz.add_argument("--no-dump", dest="dump_subjective", action="store_false")
    wz.add_argument("--dump-raw", action="store_true", default=True)
    wz.add_argument("--no-dump-raw", dest="dump_raw", action="store_false")
    wz.add_argument("--fingerprint", action="store_true", default=True)
    wz.add_argument("--no-fingerprint", dest="fingerprint", action="store_false")
    wz.add_argument("--run-id")
    wz.add_argument("--out", help="base output directory (default: runs)")
    wz.add_argument("--resume", action="store_true", default=True)
    wz.add_argument("--no-resume", dest="resume", action="store_false")
    wz.add_argument("--yes", action="store_true", default=False, help=argparse.SUPPRESS)
    wz.add_argument("--status-interval", type=float, default=5.0)
    wz.add_argument("--live-ui", choices=["off", "compact", "full", "graph", "log"], default="compact")

    df = sub.add_parser("diff", help="compare two run directories")
    df.add_argument("--a", required=True, help="first run directory")
    df.add_argument("--b", required=True, help="second run directory")
    df.add_argument("--out", help="output markdown path, default: <b>/diff.md")
    df.add_argument("--noise-band", type=float, help="label deltas within this repeatability band as tied/noise-band")

    er = sub.add_parser("export-review", help="zip useful run artefacts for GPT/Claude review")
    er.add_argument("--out", help="zip output path")
    er.add_argument("runs", nargs="+", help="run directories to include")

    rr = sub.add_parser("repeat-report", help="compare repeated runs at per-model/per-task level")
    rr.add_argument("runs", nargs="+", help="run directories to compare")
    rr.add_argument("--out", help="write markdown report path")

    sp = sub.add_parser("sensitivity-plan", help="print a diagnostic config-sensitivity sweep script")
    sp.add_argument("--run-prefix", default="v9514_config")
    sp.add_argument("--include-regex", default=r"hermes3:8b|llama3\.1:8b|qwen2\.5-coder:14b")
    sp.add_argument("--tasks", default="web_nav,needle")
    sp.add_argument("--level", choices=["smoke", "short", "full"], default="short")
    sp.add_argument("--ctx-values", default="default,4096,16384")
    sp.add_argument("--num-predict-values", default="512,2048")
    sp.add_argument("--judge", choices=["single", "panel", "off"], default="off")
    sp.add_argument("--fingerprint", action="store_true", default=False)
    sp.add_argument("--needle-max-ctx", type=int, default=None,
                    help="operator cap for generated needle sensitivity runs; defaults to 40960 when tasks include needle")

    sr = sub.add_parser("sensitivity-report", help="summarise completed config-sensitivity runs")
    sr.add_argument("runs", nargs="+", help="run directories to read")
    sr.add_argument("--out", help="write markdown report path")

    cov = sub.add_parser("coverage", help="read or update the digest-keyed coverage ledger")
    cov.add_argument("coverage_cmd", choices=["update", "show"]); cov.add_argument("--ledger", required=True); cov.add_argument("--run-dir")
    gaps = sub.add_parser("gaps", help="advisory coverage gaps; never schedules runs")
    gaps.add_argument("--ledger", required=True); gaps.add_argument("--json", action="store_true")
    gaps.add_argument("--mock", action="store_true", help="check gaps against the offline stub model list, no Ollama needed")
    dos = sub.add_parser("dossier", help="read-only composite over covered non-stale categories")
    dos.add_argument("--ledger", required=True); dos.add_argument("--runs-dir"); dos.add_argument("--out"); dos.add_argument("--weights"); dos.add_argument("--json", action="store_true")

    rnk = sub.add_parser("rankings", help="merge every run into a persistent master database and HTML model-card report")
    rnk.add_argument("--runs-dir", default="runs", help="where to read runs from")
    rnk.add_argument("--out", default="rankings", help="generated rankings database directory")
    rnk.add_argument("--rescan", action="store_true",
                     help="force every currently present run to be reread; deleted-run history remains preserved")
    rnk.add_argument("--watch", action="store_true", help="keep rescanning for run/judge sidecar changes")
    rnk.add_argument("--interval", type=float, default=5.0, help="seconds between --watch scans")
    rnk.add_argument("--exclude-model", help="non-destructively hide one model name or digest from this rankings output")
    rnk.add_argument("--include-model", help="reverse --exclude-model for one model name or digest")
    rnk.add_argument("--exclude-run", help="non-destructively hide one run ID from this rankings output")
    rnk.add_argument("--include-run", help="reverse --exclude-run for one run ID")
    rnk.add_argument("--archive-run", help="mark one run ID archived/hidden from normal rankings")
    rnk.add_argument("--unarchive-run", help="reverse --archive-run for one run ID")
    rnk.add_argument("--reason", help="generic public-safe reason saved with include/exclude/archive operations")
    rnk.add_argument("--list-excluded", action="store_true", help="print rankings exclusions and exit unless combined with --rescan")
    rnk.add_argument("--include-separate", action="store_true", help="include runs marked separate/diagnostic in this output")
    rnk.add_argument("--adopt", dest="adopt_campaign", help="adopt one validated campaign; arbitrary source directories are refused")
    rnk.add_argument("--dry-run", action="store_true", help="preview campaign adoption without canonical mutation")

    sub.add_parser("selftest", help="verify scoring logic offline")
    return p


def _main(argv=None):
    args = build_parser().parse_args(argv)
    if args.selftest or args.cmd == "selftest":
        from . import selftest
        sys.exit(selftest.run())
    cfg = Config.load(args.config)
    if args.cmd == "runtime":
        cmd_runtime(args, cfg)
    elif args.cmd == "doctor":
        cmd_doctor(args, cfg)
    elif args.cmd == "inventory":
        cmd_inventory(args, cfg)
    elif args.cmd == "capability-evidence":
        cmd_capability_evidence(args, cfg)
    elif args.cmd == "reprobe-plan":
        cmd_reprobe_plan(args, cfg)
    elif args.cmd == "reprobe-execute":
        cmd_reprobe_execute(args, cfg)
    elif args.cmd == "runtime-fit":
        cmd_runtime_fit(args, cfg)
    elif args.cmd == "plan":
        cmd_plan(args, cfg)
    elif args.cmd == "campaign":
        cmd_campaign(args, cfg)
    elif args.cmd == "run":
        cmd_run(args, cfg)
    elif args.cmd == "watch":
        sys.exit(cmd_watch(args, cfg) or 0)
    elif args.cmd == "simulate":
        cmd_simulate(args, cfg)
    elif args.cmd == "context-profile":
        cmd_context_profile(args, cfg)
    elif args.cmd == "model-cards":
        cmd_model_cards(args, cfg)
    elif args.cmd == "freeze":
        cmd_freeze(args, cfg)
    elif args.cmd == "serve":
        cmd_serve(args, cfg)
    elif args.cmd == "report":
        cmd_report(args, cfg)
    elif args.cmd == "pack-subjective":
        cmd_pack(args, cfg)
    elif args.cmd == "grade":
        cmd_grade(args, cfg)
    elif args.cmd == "judge-dumps":
        cmd_judge_dumps(args, cfg)
    elif args.cmd == "repair":
        cmd_repair(args, cfg)
    elif args.cmd == "wizard":
        cmd_wizard(args, cfg)
    elif args.cmd == "diff":
        cmd_diff(args, cfg)
    elif args.cmd == "rankings":
        cmd_rankings(args, cfg)
    elif args.cmd == "export-review":
        cmd_export_review(args, cfg)
    elif args.cmd == "repeat-report":
        cmd_repeat_report(args, cfg)
    elif args.cmd == "sensitivity-plan":
        cmd_sensitivity_plan(args, cfg)
    elif args.cmd == "sensitivity-report":
        cmd_sensitivity_report(args, cfg)
    elif args.cmd == "coverage": cmd_coverage(args, cfg)
    elif args.cmd == "gaps": cmd_gaps(args, cfg)
    elif args.cmd == "dossier": cmd_dossier(args, cfg)
    else:
        build_parser().print_help()


def main(argv=None):
    """Convert expected external llama-server failures to concise CLI exits."""
    try:
        return _main(argv)
    except Exception as exc:
        from .llama_cpp import LlamaCppError
        if isinstance(exc, LlamaCppError):
            raise SystemExit(f"llama.cpp error: {exc}") from exc
        raise


if __name__ == "__main__":
    main()
