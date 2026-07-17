import json
from datetime import datetime, timedelta, timezone

import pytest

from llm_modelbench.watch import discover_runs, resolve_run_dir


def _iso(delta: timedelta) -> str:
    """A real, relative-to-now ISO timestamp, so these tests don't depend on
    hardcoded absolute dates going stale (or landing in the future) relative
    to whenever they actually execute."""
    return (datetime.now(timezone.utc) + delta).isoformat()


def _write_status(run_dir, run_id, models_done, models_total, updated_at):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "status.json").write_text(json.dumps({
        "run_id": run_id, "models_done": models_done, "models_total": models_total,
        "updated_at": updated_at,
    }))


def test_discover_runs_reports_in_progress_and_sorts_newest_first(tmp_path):
    _write_status(tmp_path / "a", "a", 4, 4, _iso(timedelta(minutes=-10)))
    _write_status(tmp_path / "b", "b", 2, 4, _iso(timedelta(minutes=-1)))
    runs = discover_runs(tmp_path)
    assert [r["run_id"] for r in runs] == ["b", "a"]
    assert runs[0]["in_progress"] is True
    assert runs[1]["in_progress"] is False


def test_resolve_run_dir_raises_clearly_when_no_runs_exist(tmp_path):
    with pytest.raises(SystemExit, match="No runs found"):
        resolve_run_dir(tmp_path)


def test_resolve_run_dir_auto_picks_the_only_run(tmp_path):
    _write_status(tmp_path / "only", "only", 4, 4, _iso(timedelta(minutes=-5)))
    assert resolve_run_dir(tmp_path).name == "only"


def test_resolve_run_dir_prefers_the_single_in_progress_run(tmp_path):
    _write_status(tmp_path / "done", "done", 4, 4, _iso(timedelta(minutes=-20)))
    _write_status(tmp_path / "live", "live", 2, 4, _iso(timedelta(minutes=-1)))
    assert resolve_run_dir(tmp_path).name == "live"


def test_resolve_run_dir_prompts_when_genuinely_ambiguous(tmp_path):
    _write_status(tmp_path / "live1", "live1", 1, 4, _iso(timedelta(minutes=-2)))
    _write_status(tmp_path / "live2", "live2", 2, 4, _iso(timedelta(minutes=-1)))
    result = resolve_run_dir(tmp_path, choose_fn=lambda candidates: next(
        c for c in candidates if c["run_id"] == "live2"))
    assert result.name == "live2"


def test_a_long_dead_incomplete_run_is_not_flagged_in_progress_forever(tmp_path):
    """Regression guard for the real overnight-run incident: a run interrupted
    by an SSH drop / laptop sleep, which never reached models_done ==
    models_total, kept showing in_progress forever under the old
    models_done < models_total check alone. That made `llmb watch`'s
    auto-pick prefer a run that died hours or days ago over the genuinely
    active one, and the operator ended up staring at a frozen dashboard
    believing it was live. A run with no update in the last 30+ minutes must
    not be treated as in_progress, regardless of its completion ratio."""
    _write_status(tmp_path / "zombie", "zombie", 12, 33, _iso(timedelta(hours=-19)))
    runs = discover_runs(tmp_path)
    assert runs[0]["in_progress"] is False


def test_resolve_run_dir_prefers_the_genuinely_live_run_over_older_zombies(tmp_path):
    """Same real incident, exercised through resolve_run_dir: with several old,
    long-dead incomplete runs and exactly one genuinely fresh one, the fresh
    one must be auto-selected without prompting."""
    _write_status(tmp_path / "zombie1", "zombie1", 5, 33, _iso(timedelta(days=-2)))
    _write_status(tmp_path / "zombie2", "zombie2", 9, 33, _iso(timedelta(hours=-19)))
    _write_status(tmp_path / "zombie3", "zombie3", 20, 33, _iso(timedelta(hours=-40)))
    _write_status(tmp_path / "tonight", "tonight", 72, 75, _iso(timedelta(seconds=-30)))
    assert resolve_run_dir(tmp_path).name == "tonight"


def test_a_single_stale_incomplete_run_is_still_auto_picked_as_the_only_one(tmp_path):
    """When there's exactly one run on disk and it happens to be both
    incomplete and stale, it's still the only candidate, so it's still
    auto-picked (as "most recently updated", not as "in progress") rather
    than refusing to proceed. Ambiguity handling (prompting) only kicks in
    once there are 2+ candidates that can't be narrowed to one."""
    _write_status(tmp_path / "only_and_stale", "only_and_stale", 10, 33, _iso(timedelta(hours=-5)))
    assert resolve_run_dir(tmp_path).name == "only_and_stale"


from llm_modelbench.watch import _pick_active_run, _queue_step, _queue_summary_text, watch_queue


def test_pick_active_run_returns_none_when_nothing_in_progress():
    candidates = [{"run_id": "a", "in_progress": False}, {"run_id": "b", "in_progress": False}]
    assert _pick_active_run(candidates) is None


def test_pick_active_run_returns_the_in_progress_one():
    candidates = [{"run_id": "a", "in_progress": False}, {"run_id": "b", "in_progress": True}]
    assert _pick_active_run(candidates)["run_id"] == "b"


def test_queue_step_renders_and_records_a_newly_seen_run():
    candidates = [{"run_id": "model_1", "in_progress": True, "path": "p1"}]
    step = _queue_step(candidates, current_run_id=None, watched=[], idle_since=None,
                        now=1000.0, idle_grace_seconds=180.0)
    assert step["action"] == "render"
    assert step["current_run_id"] == "model_1"
    assert step["watched"] == ["model_1"]
    assert step["idle_since"] is None


def test_queue_step_does_not_re_record_the_same_run_twice():
    candidates = [{"run_id": "model_1", "in_progress": True, "path": "p1"}]
    step = _queue_step(candidates, current_run_id="model_1", watched=["model_1"],
                        idle_since=None, now=1000.0, idle_grace_seconds=180.0)
    assert step["watched"] == ["model_1"]


def test_queue_step_advances_to_the_next_model_when_the_current_one_finishes():
    """The core feature request: once model_1's run stops being in_progress
    and model_2's run appears in_progress, the queue must move to model_2
    rather than sitting frozen on model_1."""
    step1 = _queue_step([{"run_id": "model_1", "in_progress": True, "path": "p1"}],
                         current_run_id=None, watched=[], idle_since=None,
                         now=1000.0, idle_grace_seconds=180.0)
    assert step1["watched"] == ["model_1"]

    # model_1 finished (no longer in_progress), model_2 hasn't started yet: a brief gap
    step2 = _queue_step([{"run_id": "model_1", "in_progress": False, "path": "p1"}],
                         current_run_id=step1["current_run_id"], watched=step1["watched"],
                         idle_since=step1["idle_since"], now=1001.0, idle_grace_seconds=180.0)
    assert step2["action"] == "wait"
    assert step2["watched"] == ["model_1"]

    # model_2 now appears in_progress
    step3 = _queue_step([{"run_id": "model_1", "in_progress": False, "path": "p1"},
                         {"run_id": "model_2", "in_progress": True, "path": "p2"}],
                         current_run_id=step2["current_run_id"], watched=step2["watched"],
                         idle_since=step2["idle_since"], now=1005.0, idle_grace_seconds=180.0)
    assert step3["action"] == "render"
    assert step3["current_run_id"] == "model_2"
    assert step3["watched"] == ["model_1", "model_2"]


def test_queue_step_stops_after_idle_grace_period_once_something_was_watched():
    step = _queue_step([], current_run_id="model_1", watched=["model_1"],
                        idle_since=1000.0, now=1200.0, idle_grace_seconds=180.0)
    assert step["action"] == "stop"


def test_queue_step_does_not_stop_before_the_grace_period_elapses():
    step = _queue_step([], current_run_id="model_1", watched=["model_1"],
                        idle_since=1000.0, now=1050.0, idle_grace_seconds=180.0)
    assert step["action"] == "wait"


def test_queue_step_never_stops_if_nothing_was_ever_watched():
    """A brand new, empty runs dir with nothing in progress yet must wait for
    a first run to appear, not immediately declare the queue finished."""
    step = _queue_step([], current_run_id=None, watched=[], idle_since=1000.0,
                        now=100000.0, idle_grace_seconds=180.0)
    assert step["action"] == "wait"


def test_queue_summary_reports_watched_runs_and_rankings_and_logs(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    (tmp_path / "rankings").mkdir()
    (tmp_path / "rankings" / "master_summary.json").write_text(json.dumps([{"a": 1}, {"b": 2}]))
    (tmp_path / "overnight_logs").mkdir()
    text = _queue_summary_text(runs_dir, ["model_1", "model_2"])
    assert "model_1" in text and "model_2" in text
    assert "2 models" in text
    assert str(tmp_path / "overnight_logs") in text


def test_watch_queue_follows_two_models_then_stops_and_summarizes(tmp_path, capsys):
    """End-to-end (but time-injected, no real sleeping) proof of the actual
    feature: two sequential model runs, a gap in between, then the queue
    finishes. watch_queue must follow both and print a summary naming both."""
    runs_dir = tmp_path / "runs"
    (runs_dir / "model_1").mkdir(parents=True)
    (runs_dir / "model_1" / "status.json").write_text(json.dumps(
        {"run_id": "model_1", "models_done": 1, "models_total": 1, "updated_at": "x"}))

    # scripted sequence of discover_runs() results, one per loop iteration
    sequence = [
        [{"run_id": "model_1", "path": runs_dir / "model_1", "updated_at": "x", "in_progress": True}],
        [{"run_id": "model_1", "path": runs_dir / "model_1", "updated_at": "x", "in_progress": False}],
        [{"run_id": "model_2", "path": runs_dir / "model_1", "updated_at": "y", "in_progress": True}],
        [{"run_id": "model_2", "path": runs_dir / "model_1", "updated_at": "y", "in_progress": False}],
    ] + [[{"run_id": "model_2", "path": runs_dir / "model_1", "updated_at": "y", "in_progress": False}]] * 5

    clock = {"t": 1000.0}
    call_count = {"n": 0}

    def fake_discover(_runs_dir):
        i = min(call_count["n"], len(sequence) - 1)
        call_count["n"] += 1
        return sequence[i]

    def fake_sleep(_secs):
        clock["t"] += 60.0  # jump the clock forward each "tick" so the grace period elapses quickly

    def fake_time():
        return clock["t"]

    rc = watch_queue(runs_dir, clear=False, refresh=0.1, idle_grace_seconds=150.0,
                      _discover_fn=fake_discover, _sleep_fn=fake_sleep, _time_fn=fake_time)
    assert rc == 0
    out = capsys.readouterr().out
    assert "model_1" in out
    assert "model_2" in out
    assert "Queue finished" in out


class _FakeArgs:
    def __init__(self, **kw):
        self.run_id = kw.get("run_id")
        self.out = kw.get("out")
        self.once = kw.get("once", False)
        self.follow_queue = kw.get("follow_queue", None)
        self.runs_dir = kw.get("runs_dir")
        self.layout = "full"
        self.refresh = 1.0
        self.no_clear = False
        self.screen = "auto"
        self.idle_grace = 180.0


def test_bare_watch_with_no_args_defaults_to_follow_queue(monkeypatch, tmp_path):
    """The actual feature request: `./llmb-watch` with zero flags must follow
    the whole queue by default now, not sit still on one run."""
    from llm_modelbench import cli, watch as watch_mod
    called = {}
    monkeypatch.setattr(watch_mod, "watch_queue", lambda *a, **kw: called.setdefault("queue", True))
    monkeypatch.setattr(watch_mod, "watch", lambda *a, **kw: called.setdefault("single", True))
    cli.cmd_watch(_FakeArgs(runs_dir=str(tmp_path)), cfg=None)
    assert called == {"queue": True}


def test_explicit_run_id_still_gets_single_run_behavior_by_default(monkeypatch, tmp_path):
    """Naming a specific run explicitly implies you just want that one, not
    to keep following afterward -- preserves the old, expected behavior."""
    from llm_modelbench import cli, watch as watch_mod
    called = {}
    monkeypatch.setattr(watch_mod, "watch_queue", lambda *a, **kw: called.setdefault("queue", True))
    monkeypatch.setattr(watch_mod, "watch", lambda *a, **kw: called.setdefault("single", True))
    monkeypatch.setattr(cli, "_run_dir", lambda args: tmp_path)
    cli.cmd_watch(_FakeArgs(run_id="some_run", runs_dir=str(tmp_path)), cfg=None)
    assert called == {"single": True}


def test_no_follow_queue_flag_forces_old_behavior_even_with_no_run_id(monkeypatch, tmp_path):
    from llm_modelbench import cli, watch as watch_mod
    called = {}
    monkeypatch.setattr(watch_mod, "watch_queue", lambda *a, **kw: called.setdefault("queue", True))
    monkeypatch.setattr(watch_mod, "watch", lambda *a, **kw: called.setdefault("single", True))
    monkeypatch.setattr(watch_mod, "resolve_run_dir", lambda d: tmp_path)
    cli.cmd_watch(_FakeArgs(follow_queue=False, runs_dir=str(tmp_path)), cfg=None)
    assert called == {"single": True}


def test_explicit_follow_queue_flag_overrides_even_with_a_run_id(monkeypatch, tmp_path):
    from llm_modelbench import cli, watch as watch_mod
    called = {}
    monkeypatch.setattr(watch_mod, "watch_queue", lambda *a, **kw: called.setdefault("queue", True))
    monkeypatch.setattr(watch_mod, "watch", lambda *a, **kw: called.setdefault("single", True))
    cli.cmd_watch(_FakeArgs(run_id="some_run", follow_queue=True, runs_dir=str(tmp_path)), cfg=None)
    assert called == {"queue": True}


def test_once_flag_still_gets_single_frame_behavior_by_default(monkeypatch, tmp_path):
    from llm_modelbench import cli, watch as watch_mod
    called = {}
    monkeypatch.setattr(watch_mod, "watch_queue", lambda *a, **kw: called.setdefault("queue", True))
    monkeypatch.setattr(watch_mod, "watch", lambda *a, **kw: called.setdefault("single", True))
    monkeypatch.setattr(watch_mod, "resolve_run_dir", lambda d: tmp_path)
    cli.cmd_watch(_FakeArgs(once=True, runs_dir=str(tmp_path)), cfg=None)
    assert called == {"single": True}
