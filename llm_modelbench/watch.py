"""Live terminal watcher for llm-modelbench runs.

Reads runs/<run_id>/status.json and overlays live hardware telemetry. This is intentionally
separate from the benchmark runner so UI refresh cannot corrupt scoring or result writing.
"""
from __future__ import annotations

import json
import os
import sys
import shutil
import time
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from . import __version__
from .hardware import live_snapshot
from .progress import bar, pct, seconds_hms


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {"error": f"cannot read {path}: {exc}"}


def _merge_repair_child_status(status: Dict[str, Any], runs_dir: Path) -> Dict[str, Any]:
    merged = dict(status)
    child_id = str(merged.get("current_child_run") or "")
    if not child_id:
        return merged
    child_status = _load_json(Path(runs_dir) / child_id / "status.json")
    if child_status and "error" not in child_status:
        merged["child_status"] = child_status
    return merged


def _load_repair_status_for_run(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Return a repair campaign view for either a campaign or child run."""
    own_status = _load_json(run_dir / "status.json")
    if own_status and "error" not in own_status and own_status.get("status_type") == "repair":
        return _merge_repair_child_status(own_status, run_dir.parent)

    link_path = run_dir / "repair_link.json"
    if not link_path.exists():
        return None
    link = _load_json(link_path)
    plan_id = link.get("repair_plan_id")
    if not plan_id:
        return None
    status_path = run_dir.parent / f"repair_status_{plan_id}.json"
    if not status_path.exists():
        return None
    status = _load_json(status_path)
    if "error" in status:
        return None
    status.setdefault("current_child_run", run_dir.name)
    return _merge_repair_child_status(status, run_dir.parent)


def discover_runs(runs_dir: Path, stale_after_seconds: float = 1800.0) -> list:
    """Every run under runs_dir with a readable status.json, newest first.

    Each entry: {run_id, path, updated_at, in_progress}. `in_progress` is a
    best-effort heuristic (models_done < models_total AND updated recently),
    since the harness has no explicit "run finished" event, only per-model/
    per-task progress events. A run that stopped advancing (crash, SSH drop,
    laptop sleep, etc.) without ever reaching models_done == models_total
    would otherwise be flagged in_progress forever, which made `llmb watch`'s
    auto-pick logic prefer a long-dead run over the real current one.
    stale_after_seconds bounds how long "no update" still counts as active;
    default 30 minutes comfortably exceeds normal per-task cadence
    (status.json updates on every task event) without misclassifying a
    genuinely slow model as dead.
    """
    out = []
    if not runs_dir.exists():
        return out
    now = datetime.now(timezone.utc)
    for child in runs_dir.iterdir():
        status_path = child / "status.json"
        if not child.is_dir() or not status_path.exists():
            continue
        status = _load_json(status_path)
        if "error" in status:
            continue
        link_path = child / "repair_link.json"
        if link_path.exists():
            link = _load_json(link_path)
            plan_id = link.get("repair_plan_id")
            # The discoverable parent campaign is the queue authority. Keeping
            # both parent and child candidates caused queue-following to bounce
            # between the same repair and list it multiple times.
            if plan_id and (runs_dir / f"repair_status_{plan_id}.json").exists():
                continue
        models_done = status.get("models_done", 0)
        models_total = status.get("models_total", 0)
        updated_at = status.get("updated_at", "")
        age_seconds = None
        if updated_at:
            try:
                ts = datetime.fromisoformat(updated_at)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age_seconds = (now - ts).total_seconds()
            except Exception:
                age_seconds = None
        recent = age_seconds is None or age_seconds <= stale_after_seconds
        if status.get("status_type") == "repair":
            incomplete = str(status.get("phase") or "") not in {"complete", "partial", "failed"}
        elif (child / "repair_link.json").exists():
            # A repair child can retain a stale generic models_done count when
            # the authoritative parent campaign has already reached a terminal
            # state. Queue following must use the parent repair state or it can
            # render a completed campaign forever.
            repair_status = _load_repair_status_for_run(child)
            if repair_status is not None:
                incomplete = str(repair_status.get("phase") or "") not in {
                    "complete", "partial", "failed"
                }
            else:
                incomplete = bool(models_total) and models_done < models_total
        else:
            incomplete = bool(models_total) and models_done < models_total
        out.append({
            "run_id": status.get("run_id", child.name),
            "path": child,
            "updated_at": updated_at,
            "in_progress": incomplete and recent,
        })
    out.sort(key=lambda r: r["updated_at"], reverse=True)
    return out


def resolve_run_dir(runs_dir: Path, choose_fn=None) -> Path:
    """Used when the operator gave neither --run-id nor --out.

    Zero candidates: raise with a clear message. One candidate: use it,
    printing which one and why. Multiple: prefer in-progress runs; if more
    than one remains ambiguous, prompt for a selection via choose_fn (defaults
    to an interactive numbered prompt, requires a TTY).
    """
    candidates = discover_runs(runs_dir)
    if not candidates:
        raise SystemExit(
            f"No runs found under {runs_dir}. Pass --run-id <id> or --out <run_dir>."
        )
    if len(candidates) == 1:
        only = candidates[0]
        print(f"Auto-selected the only run found: {only['run_id']}", file=sys.stderr)
        return only["path"]

    in_progress = [c for c in candidates if c["in_progress"]]
    pool = in_progress or candidates
    if len(pool) == 1:
        picked = pool[0]
        reason = "in progress" if picked["in_progress"] else "most recently updated"
        print(f"Auto-selected {picked['run_id']} ({reason}).", file=sys.stderr)
        return picked["path"]

    if choose_fn is None:
        choose_fn = _prompt_choice
    return choose_fn(pool)["path"]


def _pick_active_run(candidates: list) -> Optional[Dict[str, Any]]:
    """From discover_runs() output, pick the run to actively follow right now:
    the most recently updated genuinely in-progress candidate. None if nothing
    is currently in progress (either a brief gap between models, or the whole
    queue has finished)."""
    in_progress = [c for c in candidates if c["in_progress"]]
    return in_progress[0] if in_progress else None


def _queue_step(candidates: list, current_run_id: Optional[str], watched: list,
                 idle_since: Optional[float], now: float,
                 idle_grace_seconds: float) -> Dict[str, Any]:
    """Pure decision core for one iteration of watch_queue's loop, factored out
    so it's testable without real sleeping or a live terminal.

    Returns a dict describing what to do: {"action": "render"|"wait"|"stop",
    "run": <candidate or None>, "current_run_id": ..., "watched": ...,
    "idle_since": ...}
    """
    active = _pick_active_run(candidates)
    if active is not None:
        if active["run_id"] != current_run_id and active["run_id"] not in watched:
            watched = watched + [active["run_id"]]
        return {"action": "render", "run": active, "current_run_id": active["run_id"],
                "watched": watched, "idle_since": None}

    idle_since = idle_since if idle_since is not None else now
    waited = now - idle_since
    if watched and waited >= idle_grace_seconds:
        return {"action": "stop", "run": None, "current_run_id": current_run_id,
                "watched": watched, "idle_since": idle_since, "waited": waited}
    return {"action": "wait", "run": None, "current_run_id": current_run_id,
            "watched": watched, "idle_since": idle_since, "waited": waited}


def _queue_summary_text(runs_dir: Path, watched: list) -> str:
    lines = ["", "=== Queue finished, nothing left to watch ===",
             f"{len(watched)} run(s) followed this session:"]
    for rid in watched:
        lines.append(f"  - {rid}")
    rankings_dir = runs_dir.parent / "rankings"
    summary_path = rankings_dir / "master_summary.json"
    if summary_path.exists():
        try:
            data = json.loads(summary_path.read_text())
            lines.append(f"Rankings: {len(data)} models in {summary_path}")
        except Exception:
            lines.append(f"Rankings: file present but unreadable: {summary_path}")
    else:
        lines.append("Rankings: no master_summary.json found yet.")
    logs_dir = runs_dir.parent / "overnight_logs"
    if logs_dir.exists():
        lines.append(f"Logs: {logs_dir}")
    lines.append(f"Runs directory: {runs_dir}")
    return "\n".join(lines)


def watch_queue(runs_dir: Path, *, layout: str = "full", refresh: float = 1.0,
                 clear: bool = True, screen: str = "auto",
                 idle_grace_seconds: float = 180.0,
                 _discover_fn=None, _sleep_fn=None, _time_fn=None) -> int:
    """Follow the whole queue, not just one run: once the currently-watched
    run finishes, automatically pick up whatever starts next, looping until
    no new run has appeared for idle_grace_seconds (default 3 minutes), then
    print a summary of what ran, rankings status, and log locations, and
    exit. The optional _discover_fn/_sleep_fn/_time_fn hooks exist purely for
    testing without a real clock or live status.json files.
    """
    discover = _discover_fn or discover_runs
    sleep_fn = _sleep_fn or time.sleep
    time_fn = _time_fn or time.time

    renderer = RENDERERS.get(layout, render_full)
    use_alt = bool(clear and sys.stdout.isatty() and screen in ("auto", "alternate"))
    use_clear = bool(clear and screen != "scroll")
    watched: list = []
    current_run_id: Optional[str] = None
    idle_since: Optional[float] = None
    if use_alt:
        _enter_alt_screen()
    try:
        while True:
            candidates = discover(runs_dir)
            step = _queue_step(candidates, current_run_id, watched, idle_since,
                                time_fn(), idle_grace_seconds)
            current_run_id = step["current_run_id"]
            watched = step["watched"]
            idle_since = step["idle_since"]

            if step["action"] == "stop":
                break
            if step["action"] == "render":
                run_path = step["run"]["path"]
                repair_status = _load_repair_status_for_run(run_path)
                if repair_status is not None:
                    st = repair_status
                    hw, _ = live_snapshot(None)
                    hw = st.get("simulated_hardware") or hw
                    frame = render_repair(st, hw)
                else:
                    st = _load_json(run_path / "status.json")
                    hw, _ = live_snapshot(None)
                    if "error" in st:
                        frame = st["error"]
                    elif st.get("status_type") == "context_profile":
                        frame = render_context_profile(st, hw)
                    else:
                        frame = renderer(st, hw)
            else:
                waited = int(step.get("waited", 0))
                frame = (f"Waiting for the next run to start... "
                         f"({waited}s, giving up after {int(idle_grace_seconds)}s)\n"
                         f"Watched so far: {', '.join(watched) if watched else '(none yet)'}")
            width, height = _terminal_size()
            frame = _fit_screen(frame, width, height)
            if use_clear:
                sys.stdout.write("\033[H\033[J")
            sys.stdout.write(frame + ("\n" if not use_clear else ""))
            sys.stdout.flush()
            try:
                sleep_fn(max(0.2, float(refresh)))
            except KeyboardInterrupt:
                break
    finally:
        if use_alt:
            _leave_alt_screen()
    print(_queue_summary_text(runs_dir, watched))
    return 0


def _prompt_choice(candidates: list) -> Dict[str, Any]:
    if not sys.stdin.isatty():
        listing = "\n".join(f"  {c['run_id']}" for c in candidates)
        raise SystemExit(
            "Multiple runs are active or recent and no terminal is attached to "
            f"choose one:\n{listing}\nPass --run-id <id> to select one directly."
        )
    print("Multiple runs are active or recent, pick one:", file=sys.stderr)
    for i, c in enumerate(candidates, 1):
        tag = "in progress" if c["in_progress"] else "recent"
        print(f"  {i}. {c['run_id']}  ({tag}, updated {c['updated_at']})", file=sys.stderr)
    while True:
        raw = input(f"Select 1-{len(candidates)}: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(candidates):
            return candidates[int(raw) - 1]
        print("Not a valid choice, try again.", file=sys.stderr)


def _short(s: Any, n: int = 70) -> str:
    s = str(s or "")
    return s if len(s) <= n else s[: max(0, n - 3)] + "..."


def _mb_to_gib(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    return f"{v / 1024.0:.1f}GiB"


def _finish_time(remaining: Optional[float]) -> str:
    if remaining is None:
        return "unknown"
    return (datetime.now() + timedelta(seconds=float(remaining))).strftime("%H:%M")


def _ollama_summary(base: str = "http://127.0.0.1:11434") -> str:
    try:
        with urllib.request.urlopen(base.rstrip('/') + '/api/version', timeout=1.5) as r:  # nosec B310
            version = json.loads(r.read().decode()).get('version', '?')
    except Exception:
        version = '?'
    try:
        with urllib.request.urlopen(base.rstrip('/') + '/api/ps', timeout=1.5) as r:  # nosec B310
            models = json.loads(r.read().decode()).get('models', [])
    except Exception:
        models = []
    loaded = len(models)
    processor = 'unknown'
    offload = 'unknown'
    if models:
        m = models[0]
        size = m.get('size') or 0
        vram = m.get('size_vram') or 0
        processor = '100% GPU' if size and vram >= size * 0.98 else 'mixed GPU/CPU'
        offload = 'no' if size and vram >= size * 0.98 else 'yes'
    ok = 'ok' if version != '?' else 'unknown'
    return f"Ollama {ok} | v{version} | loaded {loaded} | processor {processor} | offload {offload}"


def _common_header(st: Dict[str, Any]) -> str:
    hc = st.get('hardware_config') or {}
    gpu_name = hc.get('gpu_name') or 'GPU unknown'
    return (f"LLM MODELBENCH V{__version__}  |  RUN {st.get('run_id','?')}  |  "
            f"{str(st.get('level','?')).upper()} x{st.get('samples','?')} {st.get('sample_mode','')}  |  {gpu_name}")


def _progress_block(st: Dict[str, Any]) -> str:
    models_done, models_total = int(st.get('models_done') or 0), int(st.get('models_total') or 0)
    tasks_done, tasks_total = int(st.get('tasks_done') or 0), int(st.get('tasks_total') or 0)
    samples_done, samples_total = int(st.get('samples_done') or 0), int(st.get('samples_total') or 0)
    eta = st.get('eta') or {}
    remaining = eta.get('rolling_seconds')
    lines = [
        f"Elapsed  {seconds_hms(st.get('elapsed_seconds')):<10} ETA rolling  {seconds_hms(remaining):<10} "
        f"Finish ~{_finish_time(remaining):<5} Confidence: {st.get('eta_confidence','?')}",
        f"ETA range {seconds_hms(eta.get('low_seconds'))} - {seconds_hms(eta.get('high_seconds'))}    "
        f"Avg/model last5 {seconds_hms(eta.get('avg_model_seconds_last5'))}",
        f"Models   {models_done:02d} / {models_total:<4} excluded {int(st.get('models_skipped') or 0):<3} {bar(models_done, models_total)} {pct(models_done, models_total):>5.1f}%",
        f"Tasks    {tasks_done:03d} / {tasks_total:<4} {bar(tasks_done, tasks_total)} {pct(tasks_done, tasks_total):>5.1f}%",
        f"Samples  {samples_done:03d} / {samples_total:<4} {bar(samples_done, samples_total)} {pct(samples_done, samples_total):>5.1f}%",
    ]
    filters = st.get('filters') or []
    if filters:
        lines.append("Filters  " + _short("; ".join(filters), 110))
    return "\n".join(lines)


def _current_block(st: Dict[str, Any]) -> str:
    cur = st.get('current') or {}
    ctx = cur.get('context_length') or '?'
    off = cur.get('offload_fraction')
    off_s = 'unknown' if off is None else ('no' if float(off) <= 0.001 else f"yes {float(off)*100:.1f}%")
    return "\n".join([
        f"Model  {cur.get('model_index','?')} / {st.get('models_total','?')}   {_short(cur.get('model'), 82)}",
        f"Class  {cur.get('class','?'):<9} Size {cur.get('size_gb','?')}GB    Offload {off_s:<10} "
        f"CTX {ctx}",
        f"Task   {cur.get('task_index','?')} / {cur.get('tasks_total','?')}   {cur.get('task') or 'pending'}    "
        f"Sample {cur.get('sample_index') or '-'}/{cur.get('samples_for_task') or st.get('samples','?')}",
        f"State  {cur.get('state','?')}",
    ])


def _fmt(v: Any, default: str = 'n/a') -> str:
    return default if v is None else str(v)


def _performance_block(st: Dict[str, Any]) -> str:
    last = st.get('last_result') or {}
    completed = st.get('completed_models') or []
    last_model = completed[-1] if completed else {}
    tps = last.get('tps')
    avg_model = (st.get('eta') or {}).get('avg_model_seconds_last5')
    return "\n".join([
        f"Tok/s current    {_fmt(tps):<10} TTFT last      {_fmt(last.get('ttft_ms'))}ms",
        f"Last task        {_fmt(last.get('task')):<14} Last score     {_fmt(last.get('score')):<8} Last detail  {_short(last.get('reason'), 34)}",
        f"Last model time  {seconds_hms(last_model.get('duration_seconds')):<10} Avg/model last5 {seconds_hms(avg_model)}",
    ])


def _hardware_block(st: Dict[str, Any], hw: Dict[str, Any]) -> str:
    hc = st.get('hardware_config') or {}
    pause = hc.get('gpu_pause_temp_c') or 'auto'
    resume = hc.get('gpu_resume_temp_c') or 'auto'
    return "\n".join([
        f"GPU  {_short(hw.get('gpu_name') or hc.get('gpu_name') or 'unknown', 22):<22} Temp {hw.get('gpu_temp_c','n/a')}C / pause {pause}C / resume {resume}C     Power {hw.get('gpu_power_w','n/a')}W",
        f"GPU  Util {hw.get('gpu_util_pct','n/a')}%           VRAM {_mb_to_gib(hw.get('vram_used_mb'))} / {_mb_to_gib(hw.get('vram_total_mb'))}  {hw.get('vram_used_pct','n/a')}%",
        f"CPU  Util {hw.get('cpu_usage_pct') if hw.get('cpu_usage_pct') is not None else 'n/a'}%           Temp {hw.get('cpu_temp_c') if hw.get('cpu_temp_c') is not None else 'n/a'}C",
        f"RAM  Used {_mb_to_gib(hw.get('ram_used_mb'))} / {_mb_to_gib(hw.get('ram_total_mb'))}  {hw.get('ram_used_pct','n/a')}%     Swap {_mb_to_gib(hw.get('swap_used_mb'))}",
    ])


def _highlights_block(st: Dict[str, Any]) -> str:
    h = st.get('highlights') or {}
    best = h.get('best_quality') or {}
    fast = h.get('fastest_useful') or {}
    failed = st.get('failed_tasks') or []
    watch = failed[-1] if failed else {}
    return "\n".join([
        f"Best quality    {_short(best.get('model') or 'pending', 46):<46} {best.get('quality_avg','')}",
        f"Fastest useful  {_short(fast.get('model') or 'pending', 46):<46} {fast.get('tps_avg','')}",
        f"Watchlist       {_short(watch.get('model') or 'none', 46):<46} {watch.get('kind','')} {watch.get('task','')}",
    ])


def _failed_block(st: Dict[str, Any], limit: int = 12) -> str:
    rows = st.get('failed_tasks') or []
    if not rows:
        return "No failed, weak, or anomalous tasks recorded yet."
    out = []
    for r in rows[-limit:]:
        out.append(f"{r.get('kind','?'):<8} {_short(r.get('model'), 28):<28} {str(r.get('task')):<14} "
                   f"{str(r.get('score')):<6} {_short(r.get('reason'), 42)}")
    return "\n".join(out)


def _section(title: str, body: str) -> str:
    return f"{title}\n" + "-" * 80 + "\n" + body


def _repair_service_block(st: Dict[str, Any]) -> str:
    def verified_label(value: Optional[bool], text_val: str) -> str:
        if text_val is None or text_val == "None":
            return "n/a"
        return f"{text_val} — verified" if value else f"{text_val} — unverified"

    requested = _fmt(st.get("requested_kv_type"), "n/a")
    effective = _fmt(st.get("effective_kv_type"), "n/a")
    observed = _fmt(st.get("observed_kv_type"), "n/a")
    verified = bool(st.get("service_verified"))
    return "\n".join([
        f"Service           {_fmt(st.get('service_unit'), 'unknown')}",
        f"Requested KV      {requested}",
        f"Effective KV      {effective}",
        f"Live KV           {verified_label(verified, observed)}",
    ])


def _repair_current_block(st: Dict[str, Any]) -> str:
    child = st.get("child_status") or {}
    current = child.get("current") or {}
    last = child.get("last_result") or {}
    model = current.get("model") or st.get("current_model")
    task = current.get("task") or st.get("current_task")
    state = current.get("state") or st.get("phase")
    task_index = current.get("task_index")
    task_total = current.get("tasks_total")
    num_ctx = current.get("probe_num_ctx") or current.get("context_length") or last.get("num_ctx")
    tps = current.get("probe_tps") if current.get("probe_tps") is not None else last.get("tps")
    prompt_tps = (current.get("probe_prompt_tps") if current.get("probe_prompt_tps") is not None
                  else last.get("prompt_tps"))
    lines = [
        f"Model             {_fmt(model, 'pending')}",
        f"Task              {_fmt(task, 'pending')}",
        f"Child run         {_short(st.get('current_child_run'), 40) if st.get('current_child_run') else 'not created yet'}",
        f"Family            {_fmt(st.get('current_family'), 'n/a')}",
        f"State             {_fmt(state, 'unknown')}",
    ]
    if task_index is not None or task_total is not None:
        lines.append(f"Task progress     {_fmt(task_index, '0')} / {_fmt(task_total, '?')}")
    if current.get("probe_index") is not None:
        lines.append(
            f"Needle depth      {_fmt(current.get('probe_index'), '?')} / {_fmt(current.get('probe_total'), '?')}"
            f"  target={_fmt(current.get('probe_size'), '?')}  state={_fmt(current.get('probe_state'), '?')}"
        )
    if num_ctx is not None:
        lines.append(f"Context           {num_ctx}")
    if prompt_tps is not None or tps is not None:
        lines.append(f"Speed             prefill={_fmt(prompt_tps, 'n/a')} tok/s  decode={_fmt(tps, 'n/a')} tok/s")
    return "\n".join(lines)


_REPAIR_PHASE_LABELS = {
    "planning": "planning",
    "probing_capability": "functional capability probe",
    "running_action": "scored repair task",
    "partial": "finished with unresolved work",
    "running_standard_actions": "standard repairs (non-needle)",
    "running_standard_action": "standard repair execution",
    "running_current_kv_actions": "current/default KV — guarded needle planning",
    "running_current_kv_action": "current/default KV — guarded needle execution",
    "current_kv_complete": "current/default KV phase complete",
    "discovering_service": "discovering active Ollama service",
    "waiting_for_q8_confirmation": "waiting for q8_0 confirmation",
    "restarting_q8": "restarting Ollama at q8_0",
    "verifying_q8": "verifying q8_0",
    "running_q8_action": "q8_0 — guarded needle execution",
    "q8_complete": "q8_0 complete",
    "waiting_for_q4_confirmation": "waiting for q4_0 confirmation",
    "running_q4_action": "q4_0 — guarded needle execution",
    "q4_complete": "q4_0 complete",
    "waiting_for_restore_confirmation": "waiting for restore confirmation",
    "restoring": "restoring original service state",
    "restoring_after_error": "restoring original service state (after error)",
    "refreshing_rankings": "refreshing rankings",
    "complete": "complete",
    "failed": "failed",
}


def _repair_metric(st: Dict[str, Any], *keys: str) -> Any:
    child = st.get("child_status") or {}
    current = child.get("current") or {}
    last = child.get("last_result") or {}
    for container in (current, last, child, st):
        for key in keys:
            value = container.get(key)
            if value is not None:
                return value
    return None


def _repair_gib(value_mb: Any) -> str:
    if not isinstance(value_mb, (int, float)):
        return "n/a"
    return f"{float(value_mb) / 1024:.2f} GiB"


def _repair_pct(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def render_repair(st: Dict[str, Any], hw: Dict[str, Any]) -> str:
    """Compact evidence-first repair campaign view.

    Action completion and lifecycle progress are shown separately. This avoids
    the misleading ``0/1, 0%`` display while an action is actively restoring a
    service or refreshing rankings.
    """
    phase = str(st.get("phase") or "planning")
    phase_label = _REPAIR_PHASE_LABELS.get(phase, phase.replace("_", " "))
    actions_total = int(st.get("actions_total") or 0)
    actions_completed = int(st.get("actions_completed") or 0)
    action_index = int(st.get("current_action_index") or 0)
    is_done = phase in {"complete", "partial", "failed"}
    child = st.get("child_status") or {}
    current = child.get("current") or {}
    last = child.get("last_result") or {}

    model = current.get("model") or st.get("current_model") or "pending"
    task = current.get("task") or st.get("current_task") or "pending"
    state = current.get("state") or phase_label
    child_id = st.get("current_child_run") or st.get("last_child_run") or "not created"
    action_id = st.get("current_action_id") or "n/a"
    family = st.get("current_family") or "n/a"

    active_action = bool(action_index and not is_done and actions_completed < actions_total)
    action_state = (
        "active" if active_action
        else "done" if (is_done or (actions_total and actions_completed >= actions_total))
        else "pending"
    )
    action_position = action_index or min(actions_completed + (0 if is_done else 1), actions_total or 1)
    action_progress = actions_completed if not is_done else actions_total

    sim_step = int(st.get("simulation_step") or 0)
    sim_steps = int(st.get("simulation_steps") or 0)
    lifecycle_pct = pct(sim_step, sim_steps) if sim_steps else None

    elapsed = child.get("elapsed_seconds")
    prompt_tps = _repair_metric(st, "probe_prompt_tps", "prompt_tps")
    decode_tps = _repair_metric(st, "probe_tps", "tps")
    num_ctx = _repair_metric(st, "probe_num_ctx", "context_length", "num_ctx")
    offload = _repair_metric(st, "offload_fraction")
    score = last.get("score")
    reason = last.get("reason")
    vram_peak = _repair_metric(st, "vram_peak_mb")
    ram_delta = _repair_metric(st, "ollama_pss_delta_peak_mb", "ollama_rss_delta_peak_mb", "ram_delta_peak_mb")
    swap_delta = _repair_metric(st, "swap_delta_peak_mb")
    probe_index = current.get("probe_index")
    probe_total = current.get("probe_total")
    probe_size = current.get("probe_size")
    probe_state = current.get("probe_state")

    lines = [
        f"LLM MODELBENCH REPAIR {st.get('plan_id','?')}  |  V{__version__}  |  {phase_label.upper()}",
        "=" * 104,
        f"Plan {st.get('plan_id','?')}   outcome={st.get('outcome') or 'running'}   "
        f"recovered={int(st.get('recovered') or 0)} terminal={int(st.get('terminal_failures') or 0)} "
        f"unresolved={int(st.get('unresolved') or 0)} errors={int(st.get('errors') or 0)}",
        f"Actions {actions_completed}/{actions_total} {bar(action_progress, actions_total, 24)}   "
        f"current={action_position}/{actions_total or '?'} {action_state}",
    ]
    if st.get("simulation"):
        lines.append(
            f"Fixture {st.get('scenario')}   lifecycle={sim_step}/{sim_steps} "
            f"{bar(sim_step, sim_steps, 24)} {lifecycle_pct:5.1f}%"
        )

    lines += [
        "",
        "ACTIVE WORK",
        "-" * 104,
        f"Model     {_short(model, 92)}",
        f"Task      {_short(task, 48):<48} state={state}",
        f"Action    id={_short(action_id, 20):<20} family={family:<12} child={_short(child_id, 46)}",
    ]
    if probe_index is not None or probe_size is not None:
        lines.append(
            f"Context   tier={probe_index or '?'}/{probe_total or '?'} target={probe_size or 'n/a'} "
            f"num_ctx={num_ctx or 'n/a'} probe={probe_state or 'n/a'}"
        )
    elif num_ctx is not None:
        lines.append(f"Context   num_ctx={num_ctx}")

    lines += [
        "",
        "LIVE RESOURCES",
        "-" * 104,
        f"Timing    elapsed={seconds_hms(elapsed)}  prefill={_fmt(prompt_tps)} tok/s  "
        f"decode={_fmt(decode_tps)} tok/s  score={_fmt(score)}",
        f"GPU       {_repair_gib(hw.get('vram_used_mb'))}/{_repair_gib(hw.get('vram_total_mb'))}  "
        f"util={_fmt(hw.get('gpu_util_pct'))}% temp={_fmt(hw.get('gpu_temp_c'))}C "
        f"power={_fmt(hw.get('gpu_power_w'))}W peak={_repair_gib(vram_peak)}",
        f"Host      RAM={_repair_gib(hw.get('ram_used_mb'))}/{_repair_gib(hw.get('ram_total_mb'))}  "
        f"process_delta={_repair_gib(ram_delta)} swap_delta={_repair_gib(swap_delta)} "
        f"offload={_repair_pct(offload)}",
    ]
    if reason:
        lines.append(f"Last      {_short(reason, 94)}")

    lines += ["", "SERVICE / CAPABILITY", "-" * 104]
    requested = st.get("requested_kv_type")
    effective = st.get("effective_kv_type") or st.get("last_verified_kv_type")
    observed = st.get("observed_kv_type")
    service = st.get("service_unit")
    if is_done and st.get("restored_original_service_state"):
        lines.append(
            f"Ollama    original service state restored on {service or 'service'}; live=current/default"
        )
    elif phase == "restoring":
        lines.append(
            f"Ollama    restoring original state on {service or 'service'}; "
            f"last verified KV={effective or observed or 'current/default'}"
        )
    elif requested in (None, "current") and not service and not effective and not observed:
        lines.append("Ollama    current/default service configuration; no managed service mutation")
    elif service or requested or effective or observed:
        lines.append(
            f"Ollama    service={service or 'not mutated'} requested={requested or 'current'} "
            f"effective={effective or 'current/default'} live={observed or 'n/a'} "
            f"verified={_fmt(st.get('service_verified'))}"
        )
    else:
        lines.append("Ollama    current/default service configuration; no managed service mutation")
    if st.get("probe_state") or st.get("probe_detail"):
        lines.append(
            f"Capability state={_fmt(st.get('probe_state'))}  detail={_short(st.get('probe_detail'), 76)}"
        )
    if is_done:
        lines.append(
            f"Final     {st.get('outcome') or phase.upper()}  restored="
            f"{_fmt(st.get('restored_original_service_state'), 'not applicable')}"
        )
    else:
        lines.append("Source    parent campaign + linked child status, read atomically")
    lines += ["", "Ctrl-C detaches this watcher. It does not stop the benchmark or repair."]
    return "\n".join(lines)


def render_context_profile(st: Dict[str, Any], hw: Dict[str, Any]) -> str:
    """Dedicated context-profile view with tier and behavior-probe progress."""
    cur = st.get("current") or {}
    history = list(cur.get("probe_history") or [])
    phase = st.get("profile_phase") or (
        "needle_profile" if cur.get("task") == "needle" else cur.get("state") or "starting"
    )
    target = (st.get("hardware_config") or {}).get("context_profile_target_ctx") or 64000
    idx = int(cur.get("probe_index") or 0)
    total = int(cur.get("probe_total") or 4)
    state = cur.get("probe_state") or cur.get("state") or "starting"
    current_size = cur.get("probe_size")
    current_ctx = cur.get("probe_num_ctx") or cur.get("context_length")
    prompt_tps = cur.get("probe_prompt_tps")
    decode_tps = cur.get("probe_tps")
    validation = st.get("telemetry_validation") or {}
    behavior = st.get("behavior_probe") or validation.get("behavior_probe") or {}

    lines = [
        f"LLM MODELBENCH CONTEXT PROFILE  |  V{__version__}  |  {str(phase).replace('_',' ').upper()}",
        "=" * 104,
        f"Run {st.get('run_id','?')}   model={_short(cur.get('model') or 'pending', 58)}   target={target}",
        f"Tier {idx}/{total} {bar(max(0, idx - (0 if 'finished' in str(state) else 1)), total, 28)}  "
        f"target={current_size or 'n/a'} num_ctx={current_ctx or 'n/a'} state={state}",
        "",
        "LIVE",
        "-" * 104,
        f"Speed     prefill={_fmt(prompt_tps)} tok/s  decode={_fmt(decode_tps)} tok/s  "
        f"elapsed={seconds_hms(st.get('elapsed_seconds'))}",
        f"GPU       {_repair_gib(hw.get('vram_used_mb'))}/{_repair_gib(hw.get('vram_total_mb'))}  "
        f"util={_fmt(hw.get('gpu_util_pct'))}% temp={_fmt(hw.get('gpu_temp_c'))}C power={_fmt(hw.get('gpu_power_w'))}W",
        f"Host      RAM={_repair_gib(hw.get('ram_used_mb'))}/{_repair_gib(hw.get('ram_total_mb'))}  "
        f"swap={_repair_gib(hw.get('swap_used_mb'))}",
        "",
        "COMPLETED TIERS",
        "-" * 104,
        "Tier     num_ctx     result      prefill tok/s   decode tok/s   elapsed   VRAM peak   offload",
    ]
    if history:
        for item in history[-4:]:
            lines.append(
                f"{str(item.get('probe_size') or 'n/a'):>6}  {str(item.get('probe_num_ctx') or 'n/a'):>10}  "
                f"{str(item.get('probe_state') or 'n/a'):<10}  {_fmt(item.get('prompt_tps')):>13}  "
                f"{_fmt(item.get('tps')):>12}  {seconds_hms(item.get('elapsed_seconds')):>8}  "
                f"{_repair_gib(item.get('vram_peak_mb')):>10}  {_repair_pct(item.get('offload_fraction')):>8}"
            )
    else:
        lines.append("No tier completed yet. Tokenizer calibration or model loading may be in progress.")

    if behavior:
        lines += [
            "",
            "64K BEHAVIOR PROBE",
            "-" * 104,
            f"Status    {behavior.get('operating_status') or 'pending'}  "
            f"context={behavior.get('prompt_eval_count') or 'n/a'}  "
            f"anchors={behavior.get('all_anchors_exact')}  sequence={behavior.get('sequence_ok')}  "
            f"decode={_fmt(behavior.get('tps'))} tok/s",
            "Scope     recall/structure/repetition/speed only; agentic readiness is not assessed",
        ]
    if validation:
        lines += [
            "",
            "VALIDATION",
            "-" * 104,
            f"Passed={validation.get('passed')}  max_verified_ctx={validation.get('max_verified_ctx')}  "
            f"operating_status={validation.get('operating_status')}",
        ]
        if validation.get("critical_missing"):
            lines.append("Missing   " + ", ".join(str(x) for x in validation.get("critical_missing") or []))
    lines += ["", "Ctrl-C detaches this watcher. The profile continues in the runner terminal."]
    return "\n".join(lines)

def render_full(st: Dict[str, Any], hw: Dict[str, Any]) -> str:
    return "\n".join([
        _common_header(st), "=" * 80,
        _progress_block(st), "",
        _section("CURRENT MODEL", _current_block(st)), "",
        _section("LIVE PERFORMANCE", _performance_block(st)), "",
        _section("HARDWARE LIVE", _hardware_block(st, hw)), "",
        _section("OLLAMA SUMMARY", _ollama_summary(os.environ.get('LLM_MODELBENCH_OLLAMA_URL', 'http://127.0.0.1:11434'))), "",
        _section("HIGHLIGHTS SO FAR", _highlights_block(st)), "",
        _section("FAILED TASKS", _failed_block(st)), "",
        "ACTIONS", "-" * 80,
        "Ctrl-C quit watcher   status: runs/{}/status.json".format(st.get('run_id','?')),
    ])


def render_compact(st: Dict[str, Any], hw: Dict[str, Any]) -> str:
    return "\n".join([
        _common_header(st), "=" * 80,
        _progress_block(st).splitlines()[0],
        f"Progress models {st.get('models_done',0)}/{st.get('models_total',0)} excluded {st.get('models_skipped',0)} "
        f"tasks {st.get('tasks_done',0)}/{st.get('tasks_total',0)} "
        f"samples {st.get('samples_done',0)}/{st.get('samples_total',0)}", "",
        _section("CURRENT", _current_block(st)), "",
        _section("HARDWARE", _hardware_block(st, hw)), "",
        _section("LAST / HIGHLIGHTS", _performance_block(st) + "\n" + _highlights_block(st)),
    ])


def render_bars(st: Dict[str, Any], hw: Dict[str, Any]) -> str:
    def pctbar(label: str, val: Optional[float], right: str = "") -> str:
        if val is None:
            return f"{label:<10} n/a  " + bar(0, 100, 50, '#') + f" {right}"
        return f"{label:<10} {val:>5.1f}% {bar(int(val), 100, 50, '#')} {right}"
    return "\n".join([
        _common_header(st), "=" * 80,
        pctbar('GPU UTIL', hw.get('gpu_util_pct')),
        pctbar('VRAM', hw.get('vram_used_pct'), f"{_mb_to_gib(hw.get('vram_used_mb'))}/{_mb_to_gib(hw.get('vram_total_mb'))}"),
        pctbar('RAM', hw.get('ram_used_pct'), f"{_mb_to_gib(hw.get('ram_used_mb'))}/{_mb_to_gib(hw.get('ram_total_mb'))}"),
        pctbar('MODELS', pct(int(st.get('models_done') or 0), int(st.get('models_total') or 0)), f"{st.get('models_done')}/{st.get('models_total')}"),
        pctbar('TASKS', pct(int(st.get('tasks_done') or 0), int(st.get('tasks_total') or 0)), f"{st.get('tasks_done')}/{st.get('tasks_total')}"),
        pctbar('SAMPLES', pct(int(st.get('samples_done') or 0), int(st.get('samples_total') or 0)), f"{st.get('samples_done')}/{st.get('samples_total')}"),
        "",
        _current_block(st),
    ])


def render_failures(st: Dict[str, Any], hw: Dict[str, Any]) -> str:
    return "\n".join([_common_header(st), "=" * 80, _progress_block(st), "", _section("FAILED TASKS", _failed_block(st, 30))])


def render_hardware(st: Dict[str, Any], hw: Dict[str, Any]) -> str:
    return "\n".join([_common_header(st), "=" * 80, _section("HARDWARE LIVE", _hardware_block(st, hw)), "", _section("OLLAMA SUMMARY", _ollama_summary(os.environ.get('LLM_MODELBENCH_OLLAMA_URL', 'http://127.0.0.1:11434')))])


RENDERERS = {
    'full': render_full,
    'compact': render_compact,
    'bars': render_bars,
    'failures': render_failures,
    'hardware': render_hardware,
    'repair': render_repair,
    'context': render_context_profile,
}


def _fit_screen(text: str, width: int, height: int) -> str:
    """Clip dashboard text so a refresh redraws one terminal window without wrapping/scrolling."""
    width = max(40, int(width))
    height = max(8, int(height))
    lines = text.splitlines()
    clipped = []
    for line in lines[:height]:
        if len(line) > width:
            clipped.append(line[: max(0, width - 1)])
        else:
            clipped.append(line)
    return "\n".join(clipped)


def _terminal_size() -> Tuple[int, int]:
    sz = shutil.get_terminal_size((100, 40))
    # Keep one spare row so the cursor never forces a scroll.
    return max(40, sz.columns), max(8, sz.lines - 1)


def _enter_alt_screen() -> None:
    sys.stdout.write('\033[?1049h\033[?25l\033[2J\033[H')
    sys.stdout.flush()


def _leave_alt_screen() -> None:
    sys.stdout.write('\033[?25h\033[?1049l')
    sys.stdout.flush()


def watch(run_dir: Path, *, layout: str = 'full', refresh: float = 1.0,
          clear: bool = True, once: bool = False, screen: str = 'auto',
          exit_when_done: bool = False) -> int:
    if layout == 'interactive':
        from .interactive_nav import watch_interactive
        return watch_interactive(run_dir, refresh=refresh)
    status_path = run_dir / 'status.json'
    prev_cpu = None
    renderer = RENDERERS.get(layout, render_full)
    use_alt = bool(clear and not once and sys.stdout.isatty() and screen in ('auto', 'alternate'))
    use_clear = bool(clear and screen != 'scroll')
    if use_alt:
        _enter_alt_screen()
    try:
        while True:
            # Re-checked every refresh: a plain run never becomes a repair
            # child mid-flight, but checking once at the top would mean an
            # operator who ran `llmb-watch --run-id <child>` before the link
            # file existed yet would never promote to the parent view.
            repair_status = _load_repair_status_for_run(run_dir)
            if repair_status is not None:
                st = repair_status
                # Repair campaigns always use the parent repair renderer.
                # The normal default layout is "full", so treating every
                # recognised layout as an explicit override kept selecting the
                # generic child renderer even after repair_link.json existed.
                frame_renderer = render_repair
                hw, prev_cpu = live_snapshot(prev_cpu)
                hw = st.get("simulated_hardware") or hw
                frame = frame_renderer(st, hw)
            else:
                st = _load_json(status_path)
                hw, prev_cpu = live_snapshot(prev_cpu)
                if 'error' in st:
                    frame = st['error'] + "\n" + f"Waiting for {status_path} ..."
                elif st.get("status_type") == "context_profile":
                    frame = render_context_profile(st, hw)
                else:
                    frame = renderer(st, hw)
            width, height = _terminal_size()
            frame = _fit_screen(frame, width, height)
            if use_clear:
                sys.stdout.write('\033[H\033[J')
            sys.stdout.write(frame + ('\n' if not use_clear else ''))
            sys.stdout.flush()
            if once:
                return 0
            if exit_when_done and repair_status is not None and str(st.get("phase") or "") in {"complete", "partial", "failed"}:
                return 0
            try:
                time.sleep(max(0.2, float(refresh)))
            except KeyboardInterrupt:
                return 0
    finally:
        if use_alt:
            _leave_alt_screen()
