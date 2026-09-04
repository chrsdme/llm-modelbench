"""Anvil Stage 3B.3C -- managed ``llama-server`` materialisation.

Composes the accepted Stage 3B.2 resolution + the Stage 3B.3B lifecycle
domain into a real (or, in tests, a fake) ``llama-server`` child process:

    request  (MaterialisationRequest, from an accepted RESOLVED resolution)
        -> reuse-eligibility (structural: was the resolved candidate healthy?)
        -> artifact-identity check     (managed spawn only, unconditional)
        -> argv build                  (no shell, argv list, env overlay)
        -> endpoint allocation         (localhost, bounded candidate scan)
        -> Popen launch
        -> LaunchProcessProof capture  (/proc adapter)
        -> bounded monotonic readiness poll  (+ recipe-conformance check)
        -> OwnedRuntime + MaterialisationResult
        -> RuntimeLifecycleController  (graceful -> forced teardown)

**Authority boundaries (do not cross):**

* The resolver owns backend / GPU selection / RAM-spill permission /
  context. This module *translates* that recipe into invocation semantics;
  it never re-decides any of it. No fit calculation, no ``_recommended()``
  call, no performance search, no alternate model, no alternate backend.
* Ollama is **reuse-only** in Stage 3B.3 (amendment §23). The single place
  that rule lives is :func:`materialise`'s backend dispatch. The
  llama-server command builder does not know Ollama exists.
* GPU identity stays physical-UUID based. At the process boundary the
  selected UUIDs become ``CUDA_VISIBLE_DEVICES=GPU-<uuid>,...`` (CUDA UUID
  addressing -- verified against the installed ``llama-server`` on the
  target host). No transient ordinal is computed or persisted. A selected
  UUID that is not NVIDIA-UUID-shaped or is absent from the authoritative
  hardware inventory is a structured failure, never a silently-wrong env.

``llama-server`` flag choices are pinned against build 10326 (a bounded
``<exe> --version`` read supplies the build number -- ``--help`` does not
carry it -- and ``--help`` supplies the option tokens; recorded in
``ANVIL_PROGRESS.md``): ``--ctx-size``,
``-ngl/--n-gpu-layers`` (default ``auto``, accepts the enum ``all``),
``-fit/--fit`` (default ``on`` -- adjusts unset args to fit *live* device
memory), ``-sm/--split-mode {none,layer,row,tensor}`` (default ``layer``).
Because omitting ``-ngl`` on build 10326 makes *llama-server* decide the
GPU/CPU layer split from live VRAM, absence of a flag is **not** a spill
guarantee. The resolved ``placement_class`` is the sole placement authority:

* ``full_gpu``  -> ``-ngl all --fit off --split-mode none`` (documented
  enums, single device, fail closed on load).
* ``multi_gpu`` -> ``-ngl all --fit off --split-mode layer --tensor-split
  <resolved weights> --main-gpu 0`` (OWNER DECISION 3B.3C-OD1: the resolver
  carries a deterministic capacity-proportional inter-GPU split on
  ``ResolvedRuntime.tensor_split_weights``; this module emits it verbatim,
  never derives, defers, or rebalances it -- it is not even handed the
  topology).
* ``ram_spill`` -> ``--ctx-size <resolved concrete context> --fit on -ngl
  auto`` (+ ``--split-mode none --main-gpu 0`` for one GPU, or
  ``--split-mode layer --tensor-split <resolved weights> --main-gpu 0`` for a
  pool). OWNER DECISION 3B.3C-OD2: only after 3B.2 has resolved
  ``placement_class == "ram_spill"``, granted ``allow_ram_spill``, and a
  concrete benchmark context, llama.cpp may choose **only** the exact
  GPU-resident-vs-host layer boundary. ``allow_ram_spill`` false or an absent
  context -> ``RESOLVED_RECIPE_INCOMPLETE``, never ``--fit on``.

The ``LlamaServerCommand`` invariant ``fit == "on"`` iff ``n_gpu_layers ==
"auto"`` makes the RAM-spill delegation structurally inseparable from its
preconditions.
Before spawn, the resolved executable's ``--version`` (build pin) and
``--help`` are checked for the
launch-essential options (:data:`REQUIRED_LLAMA_SERVER_CLI_OPTIONS`); a
mismatch is :attr:`MaterialisationStatus.CLI_CONTRACT_UNSUPPORTED`, not a
launch. The bytes at ``model_path`` are hashed and matched to the resolved
artifact identity (:attr:`MaterialisationStatus.MODEL_CONTENT_MISMATCH`).
"""
from __future__ import annotations

import re
import socket
import subprocess  # noqa: S404 -- argv-list Popen, shell=False only; see _launch()
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable, Mapping, Optional, Sequence, Tuple

from .freeze import _sha256 as _sha256_file
from .hardware import GPUDevice
from .runtime_resolution import ResolvedRuntime
from .runtime_lifecycle import (
    CleanupFn,
    ForcedCleanupFailed,
    LaunchProcessProof,
    MaterialisationRequest,
    MaterialisationResult,
    RevalidateFn,
    RuntimeLifecycleController,
    RuntimeOwnership,
    materialise_owned_runtime,
    reuse_external_runtime,
)

__all__ = [
    "MaterialisationStatus",
    "ManagedMaterialisationError",
    "LlamaServerCommand",
    "EndpointCandidate",
    "ManagedMaterialisationOutcome",
    "DiagnosticSink",
    "build_llama_server_command",
    "resolve_cuda_visible_devices",
    "candidate_endpoints",
    "materialise",
    "spawn_managed_llama_server",
    "lifecycle_controller_for",
    "MANAGED_READINESS_TIMEOUT_S",
    "MANAGED_GRACEFUL_TIMEOUT_S",
    "MANAGED_FORCED_TIMEOUT_S",
    "REQUIRED_LLAMA_SERVER_CLI_OPTIONS",
    "SUPPORTED_LLAMA_SERVER_BUILD",
]

# ---------------------------------------------------------------------------
# tunables (bounded; injectable in tests)
# ---------------------------------------------------------------------------
MANAGED_READINESS_TIMEOUT_S = 90.0
MANAGED_READINESS_POLL_S = 0.25
MANAGED_GRACEFUL_TIMEOUT_S = 10.0
MANAGED_FORCED_TIMEOUT_S = 5.0
#: Bounded localhost candidate-port window for a managed server.
_ENDPOINT_BASE_PORT = 8080
_ENDPOINT_WINDOW = 64
_LOOPBACK_HOST = "127.0.0.1"

#: An NVIDIA GPU UUID as reported by ``nvidia-smi`` (``GPU-<uuid>``) or the
#: bare UUID. CUDA accepts the ``GPU-``-prefixed form in CUDA_VISIBLE_DEVICES.
_NVIDIA_UUID_RE = re.compile(
    r"^(?:GPU-)?[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


#: Launch-essential ``llama-server`` long options whose presence (and, for the
#: two placement flags, whose build-10326 default semantics) the command
#: builder depends on. If the resolved executable's ``--help`` does not
#: advertise all of these, ``spawn_managed_llama_server`` fails structurally
#: *before* spawn rather than launching an unintended configuration (audit
#: §8 -- ``--fit`` and the ``-ngl all`` enum are new build-10326 syntax; an
#: older binary silently means the opposite).
REQUIRED_LLAMA_SERVER_CLI_OPTIONS = (
    "--model",
    "--host",
    "--port",
    "--ctx-size",
    "--n-gpu-layers",
    "--fit",
    "--split-mode",
    # OWNER DECISION 3B.3C-OD1 / -OD2 -- multi-GPU deterministic split +
    # RAM-spill layer-boundary delegation launch options.
    "--tensor-split",
    "--main-gpu",
)

#: llama-server build this command builder's flag/default assumptions were
#: verified against (``llama-server --help``). Recorded so a compatibility
#: mismatch is visible, not silent.
SUPPORTED_LLAMA_SERVER_BUILD = "10326"


def _default_model_content_hasher(model_path: str) -> str:
    """Content hash of the bytes at ``model_path`` (chunked, reusing the
    repo's single file hasher). Raises ``OSError`` if the path is unreadable."""
    from pathlib import Path

    return _sha256_file(Path(model_path))


def _normalise_sha(value: str) -> str:
    """Compare model hashes regardless of an optional ``sha256:`` prefix or
    case (Ollama digests carry the prefix; a bare GGUF hash may not)."""
    v = value.strip().lower()
    if v.startswith("sha256:"):
        v = v[len("sha256:"):]
    return v


def _default_cli_contract_probe(executable_path: str) -> "frozenset[str]":
    """Return the set of long-option tokens advertised by ``<exe> --help``.

    argv-list, ``shell=False``, bounded output, short timeout. Any failure to
    run or parse yields an empty set -> the caller fails closed.

    Token presence alone cannot prove the *accepted-value* semantics the
    command builder depends on (``-ngl auto``, ``--fit on/off``, the
    ``--tensor-split`` weight-list form). Those are pinned to
    :data:`SUPPORTED_LLAMA_SERVER_BUILD`: a separate bounded ``<exe>
    --version`` read must report that build (llama-server prints
    ``version: <build> (<sha>)`` there -- ``--help`` does *not* carry it). If
    the build cannot be read or is not the supported one, the contract is
    unproven and an empty set is returned so the caller fails closed. One
    pinned-build check, not a version manager.
    """
    def _run(arg: str) -> Optional[str]:
        try:
            completed = subprocess.run(  # noqa: S603 -- argv list, shell=False
                [executable_path, arg],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=10.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return (completed.stdout or "") + "\n" + (completed.stderr or "")

    version_blob = _run("--version")
    if version_blob is None:
        return frozenset()
    build_match = re.search(
        r"(?:version|build)[:\s]+b?(\d{3,})", version_blob, re.IGNORECASE
    )
    if build_match is None or build_match.group(1) != SUPPORTED_LLAMA_SERVER_BUILD:
        return frozenset()

    help_blob = _run("--help")
    if help_blob is None:
        return frozenset()
    return frozenset(re.findall(r"--[a-z][a-z0-9-]+", help_blob))


class MaterialisationStatus(str, Enum):
    """Structured outcome of a materialisation attempt. Downstream code
    switches on this -- never on an exception string or ``pid is not None``.
    Every value has a real producer in this module."""

    #: A healthy compatible external runtime was reused; zero spawn.
    REUSED_EXTERNAL = "reused_external"
    #: A managed llama-server was spawned, proven owned, and is ready.
    SPAWNED_READY = "spawned_ready"

    # --- recipe / identity (no spawn) ----------------------------------
    #: The resolved recipe lacks launch-essential information (model artifact
    #: identity) that this module will not invent.
    RESOLVED_RECIPE_INCOMPLETE = "resolved_recipe_incomplete_for_materialisation"
    #: The model path handed to materialisation does not correspond to the
    #: artifact identity the recipe was resolved against.
    ARTIFACT_IDENTITY_MISMATCH = "artifact_identity_mismatch"
    #: A selected physical GPU UUID cannot be translated to a runtime device
    #: (not NVIDIA-UUID-shaped, or absent from the authoritative inventory).
    GPU_IDENTITY_UNTRANSLATABLE = "gpu_identity_untranslatable"
    #: The backend launch executable is not authoritatively available.
    EXECUTABLE_UNAVAILABLE = "executable_unavailable"
    #: Selected backend is Ollama and no usable external endpoint exists.
    #: Stage 3B.3 does not start/restart/mutate an Ollama daemon.
    EXTERNAL_RUNTIME_REQUIRED = "external_runtime_required"

    # --- spawn / readiness -------------------------------------------
    #: Popen itself failed (executable vanished, OSError at exec).
    SPAWN_FAILED = "spawn_failed"
    #: The child exited before readiness was established, on its own merits
    #: (not an endpoint bind race).
    PROCESS_EXITED_BEFORE_READY = "process_exited_before_ready"
    #: Readiness was not established within the bounded monotonic window.
    READINESS_TIMEOUT = "readiness_timeout"
    #: Every candidate endpoint was occupied / conflicted (incl. a lost bind
    #: race, or a foreign listener already owning a candidate port).
    ENDPOINT_CONFLICT = "endpoint_conflict"
    #: The ready listener is ours but does not match the resolved recipe
    #: (e.g. reported context differs). Retrying another port cannot help.
    WRONG_SERVICE = "wrong_service"
    #: The process launched but its ownership proof could not be established.
    OWNERSHIP_PROOF_FAILED = "ownership_proof_failed"
    #: The bytes at the supplied model path do not hash to the resolved
    #: artifact identity (or the path could not be read to prove it).
    #: Distinct from ARTIFACT_IDENTITY_MISMATCH (a caller-claim mismatch,
    #: caught earlier without touching the filesystem).
    MODEL_CONTENT_MISMATCH = "model_content_mismatch"
    #: The resolved executable's ``--help`` does not advertise a
    #: launch-essential option; command construction assumptions are not
    #: known to hold for this binary. Zero spawn.
    CLI_CONTRACT_UNSUPPORTED = "cli_contract_unsupported"


class ManagedMaterialisationError(RuntimeError):
    """Raised only for programming errors (bad argument types, a backend the
    resolver never produces). Every *expected* failure is a
    :class:`MaterialisationStatus`, not an exception."""


# ---------------------------------------------------------------------------
# command builder
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LlamaServerCommand:
    """An immutable, shell-free ``llama-server`` invocation.

    ``argv`` is a tuple where the executable, model path, host and port are
    each their own element -- never interpolated into a string, never
    ``shlex.split()`` of a built string. ``env_overlay`` carries only
    ``CUDA_VISIBLE_DEVICES`` (UUID addressing); it is applied *over* the
    parent environment at spawn.

    The argv is reconstructed from the typed fields (:func:`_render_argv`),
    so "same recipe + same executable/model/host/port -> byte-identical argv"
    is structural, not a splicing accident.
    """

    executable_path: str
    model_path: str
    host: str
    port: int
    ctx_size: Optional[int]
    split_mode: Optional[str]
    env_overlay: Mapping[str, str]
    #: ``--n-gpu-layers`` value. The builder emits either the documented enum
    #: ``"all"`` (deterministic GPU placement -- ``full_gpu`` / ``multi_gpu``),
    #: or ``"auto"`` (OWNER DECISION 3B.3C-OD2 -- ``ram_spill`` only, hands
    #: llama.cpp *only* the exact GPU-resident-vs-host layer boundary). Never a
    #: computed integer. ``None`` omits the flag.
    n_gpu_layers: Optional[str] = None
    #: ``--fit`` value (``"on"`` / ``"off"``). ``"off"`` pins llama-server's
    #: live-VRAM auto-fit *off* so placement follows the resolved recipe, not
    #: device memory at launch (deterministic placements). ``"on"`` is only
    #: reached for an owner-sanctioned ``ram_spill`` recipe. ``None`` omits it.
    fit: Optional[str] = None
    #: ``--tensor-split`` value: the resolved deterministic inter-GPU weights
    #: joined by ``","`` (OWNER DECISION 3B.3C-OD1). Present only for a
    #: layer-split placement over >= 2 GPUs. Copied from the recipe verbatim.
    tensor_split: Optional[str] = None
    #: ``--main-gpu`` value. ``0`` = ordinal zero inside the already-resolved
    #: ``CUDA_VISIBLE_DEVICES`` map -- explicitly preserves the primary GPU.
    #: Emitted for every multi-GPU launch and for a one-GPU ``ram_spill``
    #: launch (RAM-SPILL COMMAND POLICY). ``None`` omits the flag.
    main_gpu: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "env_overlay", dict(self.env_overlay))
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not 1 <= self.port <= 65535:
            raise ManagedMaterialisationError("port must be an integer 1..65535")
        if self.split_mode is not None and self.split_mode not in {"none", "layer", "row", "tensor"}:
            raise ManagedMaterialisationError("split_mode must be a llama-server split mode")
        if self.n_gpu_layers is not None and self.n_gpu_layers not in {"all", "auto"}:
            raise ManagedMaterialisationError(
                "n_gpu_layers may only be the documented enum 'all', 'auto', or "
                "None; ModelBench never emits a computed layer count"
            )
        if self.fit is not None and self.fit not in {"on", "off"}:
            raise ManagedMaterialisationError("fit must be 'on', 'off', or None")
        # OWNER DECISION 3B.3C-OD2 structural invariant: ``--fit on`` and
        # ``-ngl auto`` only ever appear together (the RAM-spill delegation),
        # and never apart. Enforced here so the "allow --fit on when spill is
        # not permitted" mutation fails at construction, not at a policy branch.
        if (self.fit == "on") != (self.n_gpu_layers == "auto"):
            raise ManagedMaterialisationError(
                "fit == 'on' iff n_gpu_layers == 'auto' (the sanctioned "
                "RAM-spill layer-boundary delegation); neither may appear alone"
            )
        if self.tensor_split is not None:
            parts = self.tensor_split.split(",")
            if len(parts) < 2 or not all(p.isdigit() and int(p) > 0 for p in parts):
                raise ManagedMaterialisationError(
                    "tensor_split must be >= 2 positive integer weights joined by ','"
                )
            if self.split_mode != "layer":
                raise ManagedMaterialisationError(
                    "tensor_split requires split_mode 'layer'"
                )
        if self.main_gpu is not None and (
            isinstance(self.main_gpu, bool)
            or not isinstance(self.main_gpu, int)
            or self.main_gpu < 0
        ):
            raise ManagedMaterialisationError("main_gpu must be a non-negative ordinal")

    @property
    def argv(self) -> Tuple[str, ...]:
        return _render_argv(self)

    def with_port(self, port: int) -> "LlamaServerCommand":
        """Return a copy bound to ``port`` -- the only environment-dependent
        argv element."""
        return LlamaServerCommand(
            executable_path=self.executable_path,
            model_path=self.model_path,
            host=self.host,
            port=port,
            ctx_size=self.ctx_size,
            split_mode=self.split_mode,
            env_overlay=self.env_overlay,
            n_gpu_layers=self.n_gpu_layers,
            fit=self.fit,
            tensor_split=self.tensor_split,
            main_gpu=self.main_gpu,
        )


def _render_argv(cmd: LlamaServerCommand) -> Tuple[str, ...]:
    argv = [
        cmd.executable_path,
        "--model",
        cmd.model_path,
        "--host",
        cmd.host,
        "--port",
        str(cmd.port),
    ]
    if cmd.ctx_size is not None:
        argv += ["--ctx-size", str(int(cmd.ctx_size))]
    if cmd.n_gpu_layers is not None:
        argv += ["--n-gpu-layers", cmd.n_gpu_layers]
    if cmd.fit is not None:
        argv += ["--fit", cmd.fit]
    if cmd.split_mode is not None:
        argv += ["--split-mode", cmd.split_mode]
    if cmd.tensor_split is not None:
        argv += ["--tensor-split", cmd.tensor_split]
    if cmd.main_gpu is not None:
        argv += ["--main-gpu", str(int(cmd.main_gpu))]
    return tuple(argv)


def resolve_cuda_visible_devices(
    selected_physical_gpu_uuids: Sequence[str],
    *,
    hardware_inventory: Sequence[GPUDevice],
) -> Tuple[Optional[str], Optional[str]]:
    """Translate the recipe's selected physical GPU UUIDs into a
    ``CUDA_VISIBLE_DEVICES`` value using **UUID addressing** (no ordinals).

    Returns ``(value, None)`` on success or ``(None, reason)`` when a
    selected UUID is not NVIDIA-UUID-shaped or is absent from the
    authoritative hardware inventory. The order of
    ``selected_physical_gpu_uuids`` is preserved verbatim (CUDA honours it --
    verified on the target host); inventory enumeration order never affects
    the result, so the same selected set always yields the same value.
    """
    if not selected_physical_gpu_uuids:
        # No GPU pinning in the recipe -> do not set the env at all.
        return None, None
    inv_uuids = {_canonical_uuid(d.uuid) for d in hardware_inventory if d.uuid}
    out = []
    for raw in selected_physical_gpu_uuids:
        if not isinstance(raw, str) or not _NVIDIA_UUID_RE.match(raw.strip()):
            return None, f"selected GPU UUID {raw!r} is not an NVIDIA GPU UUID"
        canon = _canonical_uuid(raw)
        if canon not in inv_uuids:
            return (
                None,
                f"selected GPU UUID {raw!r} is not present in the authoritative "
                f"hardware inventory",
            )
        out.append(f"GPU-{canon}")
    return ",".join(out), None


def _canonical_uuid(value: str) -> str:
    v = value.strip()
    if v.upper().startswith("GPU-"):
        v = v[4:]
    return v.lower()


def build_llama_server_command(
    request: MaterialisationRequest,
    *,
    executable_path: str,
    model_path: str,
    hardware_inventory: Sequence[GPUDevice],
    host: str = _LOOPBACK_HOST,
    port: int = _ENDPOINT_BASE_PORT,
) -> Tuple[Optional[LlamaServerCommand], Optional[MaterialisationStatus], str]:
    """Deterministically construct the ``llama-server`` command from an
    accepted resolution.

    Returns ``(command, None, "")`` on success, or ``(None, status, detail)``
    for a structured failure (untranslatable GPU identity).

    Translation rules (faithful, no search):

    * context: ``--ctx-size <requested_context>`` iff the recipe set one.
      With ``--fit off`` (see below) an omitted ``--ctx-size`` means
      llama-server takes the context from the model's own metadata -- a
      deterministic value -- rather than adjusting it to fit live VRAM.
    * GPU visibility: ``CUDA_VISIBLE_DEVICES=GPU-<uuid>,...`` over exactly
      the selected UUIDs, order preserved. Nothing else restricts device use
      (no ``--device``, which would pass transient ordinals).
    * placement authority: the resolved ``placement_class`` (one of
      ``ram_spill_preflight.PLACEMENT_LABELS``) is the *sole* placement
      decision. build 10326's ``-ngl`` default is ``auto`` and ``--fit``
      default is ``on`` -- i.e. omitting the GPU-layer flag makes
      *llama-server* decide the GPU/CPU split from live device memory. That
      would be a second placement authority, so:

      - ``full_gpu``  -> ``--n-gpu-layers all --fit off --split-mode none``.
        Every value is a documented enum, never a computed quantity. Stage
        3B.2 already proved the workload fits the selected GPU, so if the
        bytes do not actually fit, llama-server failing to load is the
        correct fail-closed outcome -- not a silent CPU spill.
      - ``multi_gpu`` -> ``--n-gpu-layers all --fit off --split-mode layer
        --tensor-split <weights> --main-gpu 0`` (OWNER DECISION 3B.3C-OD1).
        The resolver has already computed the deterministic
        capacity-proportional inter-GPU split on
        ``ResolvedRuntime.tensor_split_weights`` (GCD-reduced positive-int
        weights aligned to ``selected_physical_gpu_uuids``). This function
        only *validates* it (one positive weight per selected UUID) and
        renders it -- it never derives, defers to llama.cpp, or rebalances the
        split, and is not handed the topology. A missing / inconsistent split
        is ``RESOLVED_RECIPE_INCOMPLETE``. ``--main-gpu 0`` = ordinal zero
        inside the resolved CVD map, preserving the primary GPU.
      - ``ram_spill`` -> ``--ctx-size <resolved concrete context> --fit on
        --n-gpu-layers auto`` (+ ``--split-mode none --main-gpu 0`` for one
        GPU, or ``--split-mode layer --tensor-split <weights> --main-gpu 0``
        for a pool). OWNER DECISION 3B.3C-OD2: only after 3B.2 has resolved
        ``placement_class == "ram_spill"``, granted ``allow_ram_spill``, and a
        concrete benchmark context, llama.cpp may choose **only** the exact
        GPU-resident-vs-host layer boundary. ``allow_ram_spill`` false, an
        absent context, or (multi-GPU) a missing resolved split ->
        ``RESOLVED_RECIPE_INCOMPLETE``, never ``--fit on``. Backend, GPU pool,
        inter-GPU split, context, model artifact and sampling stay
        resolver-owned.
    """
    if not isinstance(request, MaterialisationRequest):
        raise ManagedMaterialisationError("request must be a MaterialisationRequest")
    if not (isinstance(executable_path, str) and executable_path):
        raise ManagedMaterialisationError("executable_path is required")
    if not (isinstance(model_path, str) and model_path):
        raise ManagedMaterialisationError("model_path is required")

    recipe = request.recipe
    cvd, gpu_reason = resolve_cuda_visible_devices(
        recipe.selected_physical_gpu_uuids, hardware_inventory=hardware_inventory
    )
    if gpu_reason is not None:
        return None, MaterialisationStatus.GPU_IDENTITY_UNTRANSLATABLE, gpu_reason

    placement = recipe.placement_class
    if placement == "full_gpu":
        # ``full_gpu`` is a single-device placement in every resolver path
        # (ram_spill_preflight.placement_label_for -> "full_gpu" only for
        # single_gpu_fit / candidate_single_gpu_fit, which topology_budget
        # returns with exactly one UUID; the post-discount branch labels a
        # >1-UUID pool "multi_gpu"). Enforce the invariant rather than assume
        # it -- emitting --split-mode none for a >1-UUID recipe would silently
        # override the resolved pool down to one device.
        if len(recipe.selected_physical_gpu_uuids) > 1:
            return (
                None,
                MaterialisationStatus.RESOLVED_RECIPE_INCOMPLETE,
                "resolved placement_class 'full_gpu' carries more than one "
                "selected GPU UUID; a single-device launch would silently "
                "drop GPUs the recipe selected (OWNER DECISION REQUIRED "
                "3B.3C-OD1)",
            )
        n_gpu_layers: Optional[str] = "all"
        fit: Optional[str] = "off"
        split_mode: Optional[str] = "none"
        tensor_split: Optional[str] = None
        main_gpu: Optional[int] = None
        ctx_size: Optional[int] = (
            int(recipe.requested_context)
            if recipe.requested_context is not None
            else None
        )
    elif placement == "multi_gpu":
        # OWNER DECISION 3B.3C-OD1: the resolver carries the deterministic
        # inter-GPU split. Emit it verbatim; ModelBench never derives, defers,
        # or rebalances it. --fit off / -ngl all keep placement pinned.
        split, reason = _resolved_layer_split_or_none(recipe, "multi_gpu")
        if reason is not None:
            return None, MaterialisationStatus.RESOLVED_RECIPE_INCOMPLETE, reason
        n_gpu_layers = "all"
        fit = "off"
        split_mode = "layer"
        tensor_split = split
        main_gpu = 0
        ctx_size = (
            int(recipe.requested_context)
            if recipe.requested_context is not None
            else None
        )
    elif placement == "ram_spill":
        # OWNER DECISION 3B.3C-OD2: 3B.2 has resolved placement_class=ram_spill
        # and RAM preflight passed. llama.cpp is permitted to choose ONLY the
        # exact GPU-resident-vs-host layer boundary (-ngl auto + --fit on).
        # Everything else stays resolver-owned. A concrete benchmark context
        # is mandatory -- with --fit on an absent --ctx-size is itself
        # fit-adjustable, which would let llama.cpp pick the context.
        if recipe.allow_ram_spill is not True:
            return (
                None,
                MaterialisationStatus.RESOLVED_RECIPE_INCOMPLETE,
                "resolved placement_class 'ram_spill' without an explicit "
                "allow_ram_spill permission on the recipe; a managed launch "
                "must never enable --fit on / -ngl auto without it",
            )
        if recipe.requested_context is None:
            return (
                None,
                MaterialisationStatus.RESOLVED_RECIPE_INCOMPLETE,
                "a 'ram_spill' managed launch requires a concrete resolved "
                "benchmark context (--ctx-size); with --fit on an unset "
                "context would itself be fit-adjustable (OWNER DECISION "
                "3B.3C-OD2)",
            )
        n_gpu_layers = "auto"
        fit = "on"
        ctx_size = int(recipe.requested_context)
        if len(recipe.selected_physical_gpu_uuids) > 1:
            split, reason = _resolved_layer_split_or_none(recipe, "ram_spill")
            if reason is not None:
                return None, MaterialisationStatus.RESOLVED_RECIPE_INCOMPLETE, reason
            split_mode = "layer"
            tensor_split = split
            main_gpu = 0
        else:
            # One-GPU spill: the prompt's RAM-SPILL COMMAND POLICY pins
            # --split-mode none + --main-gpu 0 (ordinal zero inside the
            # single-entry CUDA_VISIBLE_DEVICES map = the selected primary).
            split_mode = "none"
            tensor_split = None
            main_gpu = 0
    else:
        return (
            None,
            MaterialisationStatus.RESOLVED_RECIPE_INCOMPLETE,
            f"resolved placement_class {placement!r} is not a placement this "
            f"builder can faithfully translate",
        )

    env_overlay = {} if cvd is None else {"CUDA_VISIBLE_DEVICES": cvd}
    return (
        LlamaServerCommand(
            executable_path=executable_path,
            model_path=model_path,
            host=host,
            port=port,
            ctx_size=ctx_size,
            split_mode=split_mode,
            env_overlay=env_overlay,
            n_gpu_layers=n_gpu_layers,
            fit=fit,
            tensor_split=tensor_split,
            main_gpu=main_gpu,
        ),
        None,
        "",
    )


def _resolved_layer_split_or_none(
    recipe: ResolvedRuntime, placement: str
) -> Tuple[Optional[str], Optional[str]]:
    """Validate the recipe's ``tensor_split_weights`` against its selected GPU
    pool and render the ``--tensor-split`` string. Returns ``(value, None)`` on
    success or ``(None, reason)`` for a fail-closed structured failure.

    The weights are OWNER DECISION 3B.3C-OD1's authoritative resolved split.
    This function *checks* them (present, one positive integer per selected
    UUID) and joins them -- it never computes or reweights."""
    uuids = recipe.selected_physical_gpu_uuids
    weights = recipe.tensor_split_weights
    if weights is None:
        return (
            None,
            f"resolved placement_class {placement!r} selects {len(uuids)} GPUs "
            f"but the recipe carries no tensor_split_weights; ModelBench will "
            f"not derive an inter-GPU split (OWNER DECISION 3B.3C-OD1)",
        )
    if len(weights) != len(uuids):
        return (
            None,
            f"resolved tensor_split_weights {tuple(weights)!r} does not have one "
            f"weight per selected GPU ({len(uuids)}); the split is inconsistent",
        )
    if not all(isinstance(w, int) and not isinstance(w, bool) and w > 0 for w in weights):
        return (
            None,
            f"resolved tensor_split_weights {tuple(weights)!r} contains a "
            f"non-positive or non-integer weight; a selected GPU would get no "
            f"layers",
        )
    return ",".join(str(int(w)) for w in weights), None


# ---------------------------------------------------------------------------
# endpoint policy
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EndpointCandidate:
    host: str
    port: int

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


def candidate_endpoints(
    *,
    base_port: int = _ENDPOINT_BASE_PORT,
    window: int = _ENDPOINT_WINDOW,
    host: str = _LOOPBACK_HOST,
) -> Tuple[EndpointCandidate, ...]:
    """A bounded, deterministic list of localhost candidate endpoints. The
    list itself is environment-independent; which one is *used* depends on
    what is already occupied (recorded in the result)."""
    return tuple(EndpointCandidate(host, base_port + offset) for offset in range(window))


def _port_is_bindable(host: str, port: int) -> bool:
    """True iff a fresh socket can bind ``(host, port)`` right now.

    No ``SO_REUSEADDR`` -- we *want* the bind to fail if anything holds the
    port. A successful bind is not a guarantee the port stays free until the
    child binds it; that race is handled downstream by re-checking
    bindability after a child exits (a lost race -> ``ENDPOINT_CONFLICT`` for
    that candidate, try the next -- never a foreign kill).
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


# ---------------------------------------------------------------------------
# readiness + identity confirmation callables
# ---------------------------------------------------------------------------
#: Probe callable: given a base URL, return one of
#: "ready", "not_ready", "unreachable", "wrong_service".
#: The production adapter MUST catch LlamaCppError (connection refused becomes
#: "unreachable") so an uncaught raise never escapes the bounded poll loop.
ReadinessProbe = Callable[[str], str]

#: Attribution callable: given (port, pid), return one of
#: "ours"          -- the listener on port is provably our pid,
#: "foreign"       -- the listener on port is provably a different process,
#: "unestablished" -- attribution could not be established (perm-denied,
#:                    truncated scan) -- MUST NOT be read as "foreign".
PortAttribution = Callable[[int, int], str]

#: Recipe-conformance callable: given a base URL and the requested context,
#: return True iff the running server's reported context matches (or the
#: check is not applicable). A mismatch is WRONG_SERVICE.
ContextConformance = Callable[[str, Optional[int]], bool]


@dataclass(frozen=True)
class ManagedMaterialisationOutcome:
    """Everything a caller needs after a materialisation attempt."""

    status: MaterialisationStatus
    detail: str
    result: Optional[MaterialisationResult] = None
    #: The retained Popen-like object for an owned, ready runtime (so the
    #: lifecycle controller's cleanup adapter can signal it). None otherwise.
    process: object = field(default=None, repr=False)
    endpoint: Optional[str] = None
    #: Bounded tail of the child's stdout/stderr, for failure diagnostics.
    diagnostic_tail: str = ""
    #: The argv actually launched (audit); None if nothing was launched.
    launched_argv: Optional[Tuple[str, ...]] = None
    #: Anvil Stage 3B.5 -- the real env overlay applied over the parent
    #: environment at spawn (``cmd.env_overlay``; e.g.
    #: ``{"CUDA_VISIBLE_DEVICES": "GPU-<uuid>,..."}``), echoed verbatim for
    #: evidence. Never re-derived downstream from the resolved GPU UUID
    #: order -- the caller (``runtime_materialisation.materialisation_evidence``)
    #: must read this field, not guess it. ``None`` only when nothing was
    #: launched (no ``cmd`` existed yet, e.g. a pre-launch OSError).
    env_overlay: Optional[Mapping[str, str]] = None
    #: Attribution verdict at readiness time ("ours"/"unestablished"); an
    #: owned+ready runtime never returns "foreign" (that is ENDPOINT_CONFLICT).
    attribution: Optional[str] = None
    #: For a SPAWNED_READY outcome, the still-draining diagnostic sink for the
    #: owned child. The lifecycle controller closes it on teardown so the
    #: drain thread and the output pipe do not outlive the process. None for
    #: every non-owned / failed outcome (those close their sink immediately).
    diagnostic_sink: object = field(default=None, repr=False)
    #: Per-endpoint audit records retained when a managed launch cannot be
    #: materialised.  They are intentionally evidence, not retry authority.
    candidate_attempts: Tuple[Mapping[str, object], ...] = ()

    @property
    def ok(self) -> bool:
        return self.status in (
            MaterialisationStatus.REUSED_EXTERNAL,
            MaterialisationStatus.SPAWNED_READY,
        )


# ---------------------------------------------------------------------------
# top-level dispatcher -- the ONE place the Ollama reuse-only rule lives
# ---------------------------------------------------------------------------
def _resolved_candidate_was_healthy(request: MaterialisationRequest) -> bool:
    """Structural reuse-eligibility: the resolver only produces a RESOLVED
    result when it selected a candidate, and it selects one only when that
    candidate's health is ``"healthy"``. So a healthy selected candidate ==
    "a usable external runtime already exists". Deriving this from the
    request (not a caller bool) is what makes "never spawn when reuse is
    possible" structural."""
    cand = request.resolution.selected_candidate
    return cand is not None and getattr(cand, "health", None) == "healthy"


def materialise(
    request: MaterialisationRequest,
    *,
    spawn_managed: Callable[[MaterialisationRequest], ManagedMaterialisationOutcome],
    external_still_healthy: Optional[Callable[[MaterialisationRequest], bool]] = None,
) -> ManagedMaterialisationOutcome:
    """Decide, from an accepted resolution, whether to reuse an external
    runtime or materialise a managed one.

    **Hard precondition (structural, not a caller argument):** if the
    resolved candidate was ``healthy``, a usable external runtime exists and
    is reused -- ``spawn_managed`` is never reached, for any backend. This is
    the guard that "spawn llama-server even when external runtime is
    reusable" must break.

    ``external_still_healthy`` is an optional *refinement* probe (the
    resolution can be seconds stale): it may only *demote* a reuse to a
    fresh materialisation decision, never promote a spawn to a reuse. If it
    is supplied and returns ``False`` for a candidate that *was* healthy,
    the flow falls through to the no-external branch below.

    * **Any backend, external runtime usable** -> reuse it, zero spawn.
    * **Ollama, no usable external endpoint** -> structured
      :attr:`MaterialisationStatus.EXTERNAL_RUNTIME_REQUIRED`. Stage 3B.3
      never starts / restarts / mutates an Ollama daemon and never falls
      back to llama.cpp. (Amendment §23; owner-frozen for Stage 3B.3.)
    * **llama_cpp, no usable external endpoint** -> ``spawn_managed(request)``.
    """
    if not isinstance(request, MaterialisationRequest):
        raise ManagedMaterialisationError("request must be a MaterialisationRequest")

    backend = request.backend
    reuse_ok = _resolved_candidate_was_healthy(request)
    if reuse_ok and external_still_healthy is not None:
        reuse_ok = bool(external_still_healthy(request))

    if reuse_ok:
        return ManagedMaterialisationOutcome(
            status=MaterialisationStatus.REUSED_EXTERNAL,
            detail=f"reusing healthy external {backend} runtime at {request.endpoint}",
            result=reuse_external_runtime(request),
            endpoint=request.endpoint,
        )

    # Anvil Stage 3B.4: a reuse-only recipe (the resolver made no managed
    # placement decision -- Ollama per §23, or llama_cpp with no resolved
    # local GGUF) carries no launch recipe. If the external endpoint is not
    # (or no longer) reusable, this is a structured refusal -- NEVER a spawn.
    # No recipe => no launch. This guard is the plan;
    # ``spawn_managed_llama_server``'s unrecognised-placement refusal is only
    # the net.
    if not request.recipe.owned_placement:
        return ManagedMaterialisationOutcome(
            status=MaterialisationStatus.EXTERNAL_RUNTIME_REQUIRED,
            detail=(
                f"selected {backend} runtime is reuse-only (ModelBench made no "
                "managed placement decision) and the external endpoint at "
                f"{request.endpoint} is not reusable; ModelBench never spawns "
                "from a reuse-only resolution and does not fall back to a "
                "managed launch"
            ),
        )

    if backend == "ollama":
        return ManagedMaterialisationOutcome(
            status=MaterialisationStatus.EXTERNAL_RUNTIME_REQUIRED,
            detail=(
                "selected backend is Ollama and no usable external endpoint was "
                "resolved; Stage 3B.3 reuses an Ollama daemon but never starts, "
                "restarts, or reconfigures one, and does not fall back to llama.cpp"
            ),
        )

    if backend == "llama_cpp":
        return spawn_managed(request)

    raise ManagedMaterialisationError(
        f"materialise() received an unsupported backend {backend!r}; the "
        f"resolver never produces this"
    )


# ---------------------------------------------------------------------------
# managed spawn
# ---------------------------------------------------------------------------
def spawn_managed_llama_server(
    request: MaterialisationRequest,
    *,
    executable_path: Optional[str],
    model_path: Optional[str],
    model_primary_sha256: Optional[str],
    hardware_inventory: Sequence[GPUDevice],
    observe_identity: Callable[[int], Optional[LaunchProcessProof]],
    readiness_probe: ReadinessProbe,
    port_attribution: PortAttribution,
    context_conformance: ContextConformance,
    now_iso: Callable[[], str],
    popen: Optional[Callable[..., object]] = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    readiness_timeout_s: float = MANAGED_READINESS_TIMEOUT_S,
    poll_interval_s: float = MANAGED_READINESS_POLL_S,
    base_port: int = _ENDPOINT_BASE_PORT,
    endpoint_window: int = _ENDPOINT_WINDOW,
    port_bindable: Callable[[str, int], bool] = _port_is_bindable,
    content_hasher: Callable[[str], str] = _default_model_content_hasher,
    cli_contract_probe: Callable[[str], "frozenset[str]"] = _default_cli_contract_probe,
) -> ManagedMaterialisationOutcome:
    """Launch, verify, and hand back an owned ``llama-server``.

    Every argument that touches the OS is injected so the whole path is
    exercised by a fake-process integration harness without CUDA / llama.cpp
    / a model. Production wiring supplies the real adapters.
    """
    if not isinstance(request, MaterialisationRequest):
        raise ManagedMaterialisationError("request must be a MaterialisationRequest")

    # --- 1. executable authority --------------------------------------
    if not (isinstance(executable_path, str) and executable_path):
        return ManagedMaterialisationOutcome(
            status=MaterialisationStatus.EXECUTABLE_UNAVAILABLE,
            detail="llama-server launch executable is not authoritatively available",
        )

    # --- 2. model artifact authority (UNCONDITIONAL for managed spawn) --
    recipe_sha = request.recipe.model_primary_sha256
    if not (isinstance(recipe_sha, str) and recipe_sha.strip()):
        return ManagedMaterialisationOutcome(
            status=MaterialisationStatus.RESOLVED_RECIPE_INCOMPLETE,
            detail=(
                "the resolved recipe carries no model_primary_sha256; a managed "
                "llama-server launch must prove it is loading exactly the "
                "resolved artifact"
            ),
        )
    if not (isinstance(model_path, str) and model_path):
        return ManagedMaterialisationOutcome(
            status=MaterialisationStatus.RESOLVED_RECIPE_INCOMPLETE,
            detail="no local model path was supplied for the managed launch",
        )
    if not (isinstance(model_primary_sha256, str) and model_primary_sha256.strip()):
        return ManagedMaterialisationOutcome(
            status=MaterialisationStatus.ARTIFACT_IDENTITY_MISMATCH,
            detail="the supplied model path carries no content-addressed identity",
        )
    if _normalise_sha(model_primary_sha256) != _normalise_sha(recipe_sha):
        return ManagedMaterialisationOutcome(
            status=MaterialisationStatus.ARTIFACT_IDENTITY_MISMATCH,
            detail=(
                f"supplied model artifact {model_primary_sha256.strip()!r} does not "
                f"match the resolved artifact {recipe_sha.strip()!r}"
            ),
        )

    # --- 2b. model *content* authority: the claim check above only compares
    # two hash strings. Prove the bytes at ``model_path`` actually hash to the
    # resolved artifact identity before anything is launched. Managed spawn
    # only -- external reuse never reaches here.
    try:
        observed_content_sha = content_hasher(model_path)
    except OSError as exc:
        return ManagedMaterialisationOutcome(
            status=MaterialisationStatus.MODEL_CONTENT_MISMATCH,
            detail=(
                f"could not read the model bytes at {model_path!r} to prove "
                f"artifact identity: {type(exc).__name__}: {exc}"
            ),
        )
    if _normalise_sha(observed_content_sha) != _normalise_sha(recipe_sha):
        return ManagedMaterialisationOutcome(
            status=MaterialisationStatus.MODEL_CONTENT_MISMATCH,
            detail=(
                f"the bytes at {model_path!r} hash to "
                f"{_normalise_sha(observed_content_sha)!r}, not the resolved "
                f"artifact {_normalise_sha(recipe_sha)!r}"
            ),
        )

    # --- 2c. CLI contract: the command builder depends on build-10326
    # option/default semantics (``--fit off`` + ``-ngl all``). Fail closed if
    # the resolved executable does not advertise the launch-essential options
    # rather than launching an unintended configuration on an older binary.
    advertised = cli_contract_probe(executable_path)
    missing = [opt for opt in REQUIRED_LLAMA_SERVER_CLI_OPTIONS if opt not in advertised]
    if missing:
        return ManagedMaterialisationOutcome(
            status=MaterialisationStatus.CLI_CONTRACT_UNSUPPORTED,
            detail=(
                f"resolved executable {executable_path!r} does not advertise "
                f"launch-essential option(s) {missing!r}; command construction "
                f"is pinned to llama-server build {SUPPORTED_LLAMA_SERVER_BUILD} "
                f"semantics"
            ),
        )

    # --- 3. command build (GPU identity translation + placement here) --
    base_cmd, build_status, build_detail = build_llama_server_command(
        request,
        executable_path=executable_path,
        model_path=model_path,
        hardware_inventory=hardware_inventory,
        port=base_port,
    )
    if base_cmd is None:
        return ManagedMaterialisationOutcome(status=build_status, detail=build_detail)

    _popen = popen if popen is not None else _default_popen
    candidates = candidate_endpoints(base_port=base_port, window=endpoint_window)
    last_detail = "no candidate endpoint was available"
    candidate_attempts: list[Mapping[str, object]] = []

    for cand in candidates:
        if not port_bindable(cand.host, cand.port):
            last_detail = f"candidate {cand.url} is occupied"
            candidate_attempts.append({
                "endpoint": cand.url,
                "ownership_state": "not_spawned_port_unavailable",
                "launched_argv": None,
                "env_overlay": None,
                "diagnostic_tail": "",
                "reap": {"attempted": False, "ok": None, "detail": "no child launched"},
            })
            continue

        cmd = base_cmd.with_port(cand.port)
        sink = DiagnosticSink()
        try:
            proc = _launch(_popen, cmd, sink)
        except OSError as exc:
            sink.close()
            return ManagedMaterialisationOutcome(
                status=MaterialisationStatus.SPAWN_FAILED,
                detail=f"llama-server launch failed: {type(exc).__name__}: {exc}",
                diagnostic_tail=sink.tail(),
                launched_argv=cmd.argv,
                env_overlay=cmd.env_overlay,
                endpoint=cand.url,
                candidate_attempts=tuple(candidate_attempts + [{
                    "endpoint": cand.url,
                    "ownership_state": "spawn_failed",
                    "launched_argv": list(cmd.argv),
                    "env_overlay": dict(cmd.env_overlay),
                    "diagnostic_tail": sink.tail(),
                    "reap": {"attempted": False, "ok": None, "detail": "spawn failed before child existed"},
                }]),
            )

        outcome = _verify_and_own(
            request=request,
            proc=proc,
            cmd=cmd,
            endpoint=cand,
            observe_identity=observe_identity,
            readiness_probe=readiness_probe,
            port_attribution=port_attribution,
            context_conformance=context_conformance,
            now_iso=now_iso,
            monotonic=monotonic,
            sleeper=sleeper,
            readiness_timeout_s=readiness_timeout_s,
            poll_interval_s=poll_interval_s,
            sink=sink,
            port_bindable=port_bindable,
        )
        if outcome.status is MaterialisationStatus.SPAWNED_READY:
            return outcome

        # This candidate failed. Ensure the owned child is gone before the
        # next candidate (no leaked children).
        reap = _reap(proc)
        sink.close()
        attempt = {
            "endpoint": cand.url,
            "ownership_state": "not_conferred",
            "launched_argv": list(cmd.argv),
            "env_overlay": dict(cmd.env_overlay),
            "diagnostic_tail": outcome.diagnostic_tail,
            "reap": reap,
        }
        candidate_attempts.append(attempt)
        outcome = replace(
            outcome,
            endpoint=cand.url,
            candidate_attempts=tuple(candidate_attempts),
        )
        if outcome.status is MaterialisationStatus.ENDPOINT_CONFLICT:
            last_detail = outcome.detail
            continue
        # Any other failure is terminal -- another port cannot help.
        return outcome

    return ManagedMaterialisationOutcome(
        status=MaterialisationStatus.ENDPOINT_CONFLICT,
        detail=(
            f"no free managed endpoint in {candidates[0].url}..+{endpoint_window}: "
            f"{last_detail}"
        ),
        candidate_attempts=tuple(candidate_attempts),
    )


def _verify_and_own(
    *,
    request: MaterialisationRequest,
    proc,
    cmd: LlamaServerCommand,
    endpoint: EndpointCandidate,
    observe_identity,
    readiness_probe: ReadinessProbe,
    port_attribution: PortAttribution,
    context_conformance: ContextConformance,
    now_iso,
    monotonic,
    sleeper,
    readiness_timeout_s: float,
    poll_interval_s: float,
    sink: "DiagnosticSink",
    port_bindable: Callable[[str, int], bool],
) -> ManagedMaterialisationOutcome:
    def _exit_reason() -> ManagedMaterialisationOutcome:
        """Discriminate a lost bind race (retry) from a genuine early death
        (terminal). If the port is *no longer* bindable the child lost the
        race to a foreign binder; if it is bindable again the child died on
        its own merits."""
        if not port_bindable(endpoint.host, endpoint.port):
            return ManagedMaterialisationOutcome(
                status=MaterialisationStatus.ENDPOINT_CONFLICT,
                detail=(
                    f"candidate {endpoint.url} was taken between the bind check "
                    f"and the child bind (lost race); trying the next candidate"
                ),
                diagnostic_tail=sink.tail(),
                launched_argv=cmd.argv,
                env_overlay=cmd.env_overlay,
            )
        return ManagedMaterialisationOutcome(
            status=MaterialisationStatus.PROCESS_EXITED_BEFORE_READY,
            detail=f"llama-server exited (rc={proc.returncode}) before readiness",
            diagnostic_tail=sink.tail(),
            launched_argv=cmd.argv,
            env_overlay=cmd.env_overlay,
        )

    # --- ownership proof BEFORE declaring anything --------------------
    pid = getattr(proc, "pid", None)
    if not isinstance(pid, int) or pid <= 0:
        return ManagedMaterialisationOutcome(
            status=MaterialisationStatus.OWNERSHIP_PROOF_FAILED,
            detail="launched process has no usable PID",
            diagnostic_tail=sink.tail(),
            launched_argv=cmd.argv,
            env_overlay=cmd.env_overlay,
        )
    if proc.poll() is not None:
        return _exit_reason()
    launch_proof = observe_identity(pid)
    if launch_proof is None:
        return ManagedMaterialisationOutcome(
            status=MaterialisationStatus.OWNERSHIP_PROOF_FAILED,
            detail="could not establish a process-identity proof for the launched pid",
            diagnostic_tail=sink.tail(),
            launched_argv=cmd.argv,
            env_overlay=cmd.env_overlay,
        )

    # --- bounded monotonic readiness poll ----------------------------
    # Two independent bounds: a monotonic-time deadline AND a hard cap on the
    # number of poll iterations. Either alone terminates the loop; together
    # they mean a broken clock or a probe that never advances time still
    # cannot spin forever.
    deadline = monotonic() + max(0.0, float(readiness_timeout_s))
    interval = min(max(0.001, float(poll_interval_s)), max(0.001, float(readiness_timeout_s)))
    max_attempts = max(1, int(float(readiness_timeout_s) / interval) + 2)
    attempts = 0
    while True:
        attempts += 1
        if proc.poll() is not None:
            return _exit_reason()
        verdict = readiness_probe(endpoint.url)
        if verdict == "ready":
            break
        if verdict == "wrong_service":
            # An endpoint answered on our candidate port with a non-llama
            # service -> the port is taken by a foreign process. Retry.
            return ManagedMaterialisationOutcome(
                status=MaterialisationStatus.ENDPOINT_CONFLICT,
                detail=f"{endpoint.url} is answered by a non-llama-server; trying next",
                diagnostic_tail=sink.tail(),
                launched_argv=cmd.argv,
                env_overlay=cmd.env_overlay,
            )
        if monotonic() >= deadline or attempts >= max_attempts:
            return ManagedMaterialisationOutcome(
                status=MaterialisationStatus.READINESS_TIMEOUT,
                detail=(
                    f"llama-server did not become ready within {readiness_timeout_s:g}s "
                    f"({attempts} poll attempts)"
                ),
                diagnostic_tail=sink.tail(),
                launched_argv=cmd.argv,
                env_overlay=cmd.env_overlay,
            )
        sleeper(interval)

    # --- prove the ready endpoint is OUR process -------------------
    attribution = port_attribution(endpoint.port, pid)
    if attribution == "foreign":
        # A foreign process owns the ready listener -> endpoint conflict,
        # retry the next candidate (NOT wrong_service -- that is reserved for
        # "the listener is ours but fails recipe conformance").
        return ManagedMaterialisationOutcome(
            status=MaterialisationStatus.ENDPOINT_CONFLICT,
            detail=(
                f"the ready listener on {endpoint.url} is provably a different "
                f"process; trying the next candidate"
            ),
            diagnostic_tail=sink.tail(),
            launched_argv=cmd.argv,
            env_overlay=cmd.env_overlay,
        )
    # "ours" or "unestablished" -> proceed, preserving which it was.

    # --- independent recipe conformance -------------------------
    if not context_conformance(endpoint.url, request.recipe.requested_context):
        return ManagedMaterialisationOutcome(
            status=MaterialisationStatus.WRONG_SERVICE,
            detail=(
                f"the ready llama-server at {endpoint.url} does not match the "
                f"resolved context ({request.recipe.requested_context})"
            ),
            diagnostic_tail=sink.tail(),
            launched_argv=cmd.argv,
            env_overlay=cmd.env_overlay,
        )

    result = materialise_owned_runtime(
        request,
        launch_proof=launch_proof,
        launched_at=now_iso(),
    )
    return ManagedMaterialisationOutcome(
        status=MaterialisationStatus.SPAWNED_READY,
        detail=f"managed llama-server ready at {endpoint.url} (pid {pid})",
        result=result,
        process=proc,
        endpoint=endpoint.url,
        diagnostic_tail=sink.tail(),
        launched_argv=cmd.argv,
        env_overlay=cmd.env_overlay,
        attribution=attribution,
        diagnostic_sink=sink,
    )


# ---------------------------------------------------------------------------
# stdout/stderr: bounded ring buffer, actively drained, cleaned on close
# ---------------------------------------------------------------------------
class DiagnosticSink:
    """A bounded diagnostic buffer for a child's merged stdout/stderr.

    A background thread drains the child's output pipe into a fixed-size ring
    buffer so the child can never block on a full OS pipe, and only the last
    ``max_bytes`` are retained (no unbounded log growth). :meth:`close`
    stops the drainer and releases resources.
    """

    def __init__(self, max_bytes: int = 64 * 1024) -> None:
        import threading

        self._max = max_bytes
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._thread = None
        self._stream = None

    def attach(self, stream) -> None:
        import threading

        self._stream = stream

        def _drain() -> None:
            try:
                while True:
                    chunk = stream.read(4096)
                    if not chunk:
                        return
                    with self._lock:
                        self._buf.extend(chunk)
                        if len(self._buf) > self._max:
                            del self._buf[: len(self._buf) - self._max]
            except (OSError, ValueError):
                return

        self._thread = threading.Thread(
            target=_drain, name="llama-server-diag", daemon=True
        )
        self._thread.start()

    def tail(self) -> str:
        with self._lock:
            return bytes(self._buf).decode("utf-8", "replace")

    def close(self) -> None:
        stream = self._stream
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Popen -- argv list, never shell
# ---------------------------------------------------------------------------
def _default_popen(argv, **kwargs):
    return subprocess.Popen(argv, **kwargs)  # noqa: S603 -- argv list, shell=False


def _launch(popen, cmd: LlamaServerCommand, sink: DiagnosticSink):
    """Start the child with an argv list (no shell) and an env overlay, and
    attach the bounded diagnostic drainer."""
    import os as _os

    env = dict(_os.environ)
    env.update(cmd.env_overlay)
    proc = popen(
        list(cmd.argv),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        shell=False,
        close_fds=True,
    )
    if getattr(proc, "stdout", None) is not None:
        sink.attach(proc.stdout)
    return proc


def _reap(proc) -> Mapping[str, object]:
    """Best-effort ensure a failed candidate child is not left running.

    Only reached for a child WE just launched that has NOT been conferred an
    OwnedRuntime -- direct termination is safe. A child that already exited
    is simply waited on."""
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=MANAGED_GRACEFUL_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=MANAGED_FORCED_TIMEOUT_S)
        else:
            proc.wait(timeout=1.0)
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        return {"attempted": True, "ok": False, "detail": f"{type(exc).__name__}: {exc}"}
    return {"attempted": True, "ok": True, "detail": "child reaped"}


# ---------------------------------------------------------------------------
# lifecycle-controller wiring for an owned managed runtime
# ---------------------------------------------------------------------------
def lifecycle_controller_for(
    outcome: ManagedMaterialisationOutcome,
    *,
    observe_identity: Callable[[int], Optional[LaunchProcessProof]],
    terminate: Callable[..., str],
    graceful_timeout_s: float = MANAGED_GRACEFUL_TIMEOUT_S,
    forced_timeout_s: float = MANAGED_FORCED_TIMEOUT_S,
) -> RuntimeLifecycleController:
    """Build a :class:`RuntimeLifecycleController` for a materialisation
    outcome.

    * REUSED_EXTERNAL -> controller with no cleanup authority (no callbacks).
    * SPAWNED_READY -> controller whose ``revalidate_fn`` re-observes the
      ``/proc`` identity and whose ``cleanup_fn`` runs a scoped graceful ->
      forced termination of exactly the retained child, re-proving ownership
      before *each* signal. A process that survives both signals within
      their bounded windows raises :class:`ForcedCleanupFailed`, which the
      controller maps to :attr:`CleanupOutcome.FORCED_FAILED`.
    """
    result = outcome.result
    if result is None:
        raise ManagedMaterialisationError("outcome carries no MaterialisationResult")

    if result.ownership is RuntimeOwnership.EXTERNAL_REUSED:
        return RuntimeLifecycleController(result)

    proc = outcome.process
    if proc is None:
        raise ManagedMaterialisationError("owned outcome carries no retained process")

    def _revalidate(owned) -> Optional[LaunchProcessProof]:
        return observe_identity(owned.launch_proof.pid)

    sink = outcome.diagnostic_sink

    def _cleanup(owned) -> None:
        def _still_ours() -> bool:
            observed = observe_identity(owned.launch_proof.pid)
            return observed is not None and owned.launch_proof.revalidation_matches(observed)

        try:
            verdict = terminate(
                proc,
                graceful_timeout_s=graceful_timeout_s,
                forced_timeout_s=forced_timeout_s,
                revalidate=_still_ours,
            )
        finally:
            # Close the diagnostic drainer + output pipe regardless of the
            # teardown outcome -- they must not outlive the process.
            if sink is not None:
                sink.close()
        if verdict == "survived":
            raise ForcedCleanupFailed(
                "managed llama-server survived graceful and forced termination "
                "within their bounded windows"
            )

    revalidate_fn: RevalidateFn = _revalidate
    cleanup_fn: CleanupFn = _cleanup
    return RuntimeLifecycleController(
        result, cleanup_fn=cleanup_fn, revalidate_fn=revalidate_fn
    )
