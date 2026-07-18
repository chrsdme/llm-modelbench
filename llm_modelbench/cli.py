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
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from . import __version__
from .config import Config
from .classify import classify_model, families_for, size_gb
from .filters import parse_task_ids


def _ollama_port(url: str) -> int:
    parsed = urlparse(url if "://" in url else f"http://{url}")
    try:
        return int(parsed.port or 11434)
    except ValueError as exc:
        raise SystemExit(f"invalid Ollama URL port in {url!r}") from exc


def _client(args, cfg: Config):
    if getattr(args, "mock", False):
        from .ollama import MockClient
        return MockClient(cfg.ollama_url, cfg.seed, cfg.temperature, cfg.request_timeout)
    from .ollama import OllamaClient
    return OllamaClient(cfg.ollama_url, cfg.seed, cfg.temperature, cfg.request_timeout)


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
    items = []
    for model_row in rows:
        name = model_row.get("name", "")
        profile = interrogate_model(client, name, functional=bool(getattr(args, "auto", False)))
        families = profile.get("supported_families") or []
        items.append({"name": name,
                      "class": classify_model(name, profile.get("declared_capabilities"), families),
                      "size_gb": size_gb(model_row), "families": families,
                      "declared_capabilities": profile.get("declared_capabilities") or [],
                      "capability_warnings": profile.get("warnings") or [],
                      "will_offload": size_gb(model_row) > cfg.vram_budget_gb})
    items.sort(key=lambda x: (x["class"], -x["size_gb"]))
    if args.json:
        print(json.dumps(items, indent=2)); return
    print(f"VRAM budget: {cfg.vram_budget_gb}GB\n")
    for it in items:
        flag = "OFFLOAD" if it["will_offload"] else "fits"
        print(f"{it['class']:<11} {it['size_gb']:>6}GB  {flag:<8} {it['name']}")




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


def _plan_for_args(args, cfg, client, *, selected_models=None, capability_profiles=None):
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
    client = _client(args, cfg)
    out_dir = _run_dir(args)
    rankings_dir = _ranking_dir_for(args, run_id=out_dir.name)
    _write_run_ranking_scope(out_dir, args, rankings_dir=rankings_dir)
    task_ids = _task_ids(args)
    selected_models = getattr(args, "_selected_models", None)
    if selected_models is None:
        selected_models = _resolve_model_selection(args, client)
    capability_profiles = getattr(args, "_capability_profiles", None)
    plan = getattr(args, "_accepted_plan", None) or _plan_for_args(
        args, cfg, client, selected_models=selected_models, capability_profiles=capability_profiles
    )
    _require_host_code_opt_in(args, plan)
    _confirm_plan(args, plan)
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
                   auto_probe=bool(getattr(args, "auto", False)))
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


def cmd_campaign(args, cfg):
    """Thin compatibility layer: existing runners receive a normal nested run dir."""
    from . import campaign, planner
    if args.campaign_cmd == "status":
        paths, manifest = _campaign_paths_or_exit(args.campaign_id)
        print(json.dumps({"campaign_id": manifest.campaign_id, "state": manifest.state,
                          "resume_state": manifest.resume_state, "root": str(paths.root)}, indent=2))
        return
    if args.campaign_cmd == "resume":
        paths, manifest = _campaign_paths_or_exit(args.campaign_id)
        if manifest.state != "interrupted" or not manifest.resume_state:
            raise SystemExit("campaign resume requires an interrupted campaign with a recorded resume phase")
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
        else:
            models = [value.strip() for value in (args.models or "").split(";") if value.strip()]
            paths, manifest = campaign.create_campaign(args.campaign_id, models=models, level=args.level, version=__version__)
        client = _client(args, cfg)
        selected = _resolve_model_selection(args, client)
        plan = _plan_for_args(args, cfg, client, selected_models=selected)
        campaign.write_campaign_plan(paths, plan, inventory=client.tags(), capabilities=plan.get("capability_profiles") or {}, configuration={"level": args.level, "models": args.models, "judge_policy": getattr(args, "judge", "off"), "samples": args.samples, "think": args.think, "ctx": args.ctx, "num_predict": args.num_predict})
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
            manifest = campaign.transition(paths, manifest, "generating")
        elif manifest.state != "generating":
            raise SystemExit(f"campaign {args.campaign_id!r} cannot run from state {manifest.state!r}")
        lock = campaign.acquire_lock(paths, operation="campaign-run", phase="generating")
        try:
            client = _client(args, cfg)
            selected = _resolve_model_selection(args, client)
            accepted_plan = _plan_for_args(args, cfg, client, selected_models=selected)
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
                cohort = [{"name": row.get("model"), "digest": row.get("model_digest_resolved")} for row in rows]
                candidates = [{"name": item.get("name"), "digest": item.get("digest"), "supported_families": ["text"], "priority": 0, "calibrated": False} for item in inventory]
                judge = campaign.select_campaign_judge(candidates, cohort)
                selection = {"eligible": len(eligible), "cohort": cohort, "machine_judged_provisional": True, "judge": judge}
                campaign._atomic_write_text(paths.judge_dir / "judge_selection.json", json.dumps(selection, indent=2, sort_keys=True))
                if judge:
                    from . import judge_dumps
                    judged = judge_dumps.judge_run(client, paths.primary_dir, judge_model=judge["name"], judge_mode="single")
                    if (paths.primary_dir / "judge_results.jsonl").exists():
                        __import__("shutil").copy2(paths.primary_dir / "judge_results.jsonl", paths.judge_results)
                    campaign._atomic_write_text(paths.judge_summary, json.dumps({**judged, "selection": selection}, indent=2, sort_keys=True))
                    rows = judge_dumps.apply_judgements(paths.primary_dir, rows)
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
    client = _client(args, cfg)

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
    rankings_dir = _ranking_dir_for(args, run_id=f"repair_{plan.plan_id}")
    ranking_scope = "separate" if getattr(args, "separate_ranking", False) else "canonical"
    if args.kv_cascade:
        from .ollama_service import OllamaServiceController
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
                preflight = OllamaServiceController(
                    "ollama.service", port=port, auto_confirm=True,
                )
                preflight.verify_noninteractive_sudo_ready()
                print("Preflight passed: passwordless sudo is ready for the fallback phases.\n")
            if args.ollama_service == "auto":
                discovery_guard = OllamaServiceController(
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
                controller = OllamaServiceController.for_active_service(
                    port=port, force_password_prompt=force_password,
                    auto_confirm=auto_confirm, event_callback=record_service_event,
                    warn_fn=lambda message: record_service_event({
                        "phase": "discover", "warning": message,
                    }),
                )
            else:
                controller = OllamaServiceController(
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
    plan, options = wizard.interactive_plan(
        client, cfg,
        initial_level=args.level,
        judge_mode=args.judge,
        initial_categories=_categories(args),
        initial_task_ids=_task_ids(args),
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

def _quality_by_digest_from_ledger(ledger: dict) -> dict:
    """Load each ledger-referenced run and map aggregate category quality by digest."""
    from .aggregate import aggregate
    from .tasks import TASKS
    difficulty = {task.id: task.difficulty for task in TASKS}
    run_dirs = {entry.get("out_dir") for ledger_entry in ledger.values()
                for entry in ledger_entry.get("categories", {}).values() if entry.get("out_dir")}
    quality_by_digest = {}
    for out_dir in run_dirs:
        run = Path(out_dir)
        raw_path, identities_path = run / "raw_results.jsonl", run / "model_identities.json"
        if not raw_path.exists() or not identities_path.exists():
            continue
        rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
        identities = json.loads(identities_path.read_text())
        _, per_cat = aggregate(rows, cfg_weights_for(run), difficulty)
        for category, ranked in per_cat.items():
            for model_name, quality in ranked:
                digest = (identities.get(model_name) or {}).get("digest")
                if digest:
                    quality_by_digest.setdefault(digest, {})[category] = quality
    return quality_by_digest

def cmd_dossier(args, cfg):
    from .coverage import load_ledger
    from .dossier import DEFAULT_CATEGORY_WEIGHTS, composite_score, validate_weights
    from .weights_override import parse_weight_overrides
    from .tasks import TASKS
    weights = parse_weight_overrides(args.weights, DEFAULT_CATEGORY_WEIGHTS) if args.weights else DEFAULT_CATEGORY_WEIGHTS
    validate_weights(weights); ledger = load_ledger(Path(args.ledger)); out = {}; quality_by_digest = _quality_by_digest_from_ledger(ledger)
    for digest, entry in ledger.items():
        out[digest] = composite_score(digest, ledger, quality_by_digest.get(digest, {}), weights, TASKS) | {"names_seen": entry.get("names_seen", [])}
    text=json.dumps(out, indent=2)
    if args.out: Path(args.out).write_text(text)
    if args.json or not args.out: print(text)


# Shared run/plan arguments. Keep the wizard/doctor simple and the core CLI scriptable.
def _add_run_filters(
    r, *, include_model_selection: bool = True, include_auto: bool = True,
    auto_default: bool = False,
):
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
    p.add_argument("--selftest", action="store_true", help="run offline scorer tests and exit")
    sub = p.add_subparsers(dest="cmd")

    inv = sub.add_parser("inventory", help="list local models")
    inv.add_argument("--json", action="store_true")
    inv.add_argument("--mock", action="store_true", help="use offline stub model list")
    inv.add_argument("--auto", action="store_true", help="also run functional capability probes")

    doc = sub.add_parser("doctor", help="preflight environment, import path, Ollama, GPU, disk")
    doc.add_argument("--json", action="store_true")

    pl = sub.add_parser("plan", help="show active models, skipped models, tasks, samples, and rough ETA without running")
    _add_run_filters(pl)
    pl.add_argument("--judge", choices=["single", "panel", "off"], default="off",
                    help="subjective scoring mode used for sample planning; default: off")
    pl.add_argument("--json", action="store_true")

    camp = sub.add_parser("campaign", help="manage isolated campaign workspaces")
    camp_sub = camp.add_subparsers(dest="campaign_cmd", required=True)
    camp_status = camp_sub.add_parser("status", help="show campaign lifecycle state")
    camp_status.add_argument("campaign_id")
    camp_resume = camp_sub.add_parser("resume", help="resume the exact recorded interrupted campaign phase")
    camp_resume.add_argument("campaign_id")
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
    camp_plan = camp_sub.add_parser("plan", help="create an isolated campaign plan")
    camp_plan.add_argument("--campaign-id", required=True)
    _add_run_filters(camp_plan)
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
    camp_run.add_argument("--unattended-safe", action="store_true", help="write terminal readiness and review package without host mutation")

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
                    help="systemd unit managed by --restart-ollama; default auto discovers the unit owning the configured Ollama port")
    rp.add_argument("--keep-final-kv", action="store_true",
                    help="do not restore the original Ollama service drop-in after the cascade")
    rp.add_argument("--reuse-sudo-credentials", action="store_true",
                    help="do not invalidate sudo's cached timestamp before each privileged phase")
    rp.add_argument("--auto-confirm", action="store_true",
                    help="fully unattended apply mode for --restart-ollama --kv-cascade: implies --yes, skips "
                         "typed DISCOVER/VERIFY/RESTART confirmations, and uses sudo -n only. Requires a scoped "
                         "NOPASSWD sudoers rule for the exact commands in docs/auto_confirm_sudoers.md. Does "
                         "not store or read a password. Off by default.")
    rp.add_argument("--force", action="store_true", help="allow a previously recorded repair action to be planned again")
    rp.add_argument("--live-ui", choices=["off", "compact", "full", "log"], default="compact",
                    help="inline repair-aware dashboard for child runs; detached llmb-watch remains supported")
    rp.add_argument("--mock", action="store_true", help="offline deterministic repair pipeline test")
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


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.selftest or args.cmd == "selftest":
        from . import selftest
        sys.exit(selftest.run())
    cfg = Config.load(args.config)
    if args.cmd == "doctor":
        cmd_doctor(args, cfg)
    elif args.cmd == "inventory":
        cmd_inventory(args, cfg)
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


if __name__ == "__main__":
    main()
