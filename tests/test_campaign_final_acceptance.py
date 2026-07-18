import json
import zipfile
from collections import Counter
from pathlib import Path

from llm_modelbench import campaign, cli


def test_cli_forced_mock_campaign_runs_full_terminal_lifecycle(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = {
        "rules": [
            {"contains": "Extract to JSON", "responses": [{"text": '{"server":"API-01","ip":"192.168.1.10","status":"Critical"}'}]},
            {"contains": "Which commit hash introduced", "responses": [{"text": "0000000"}]},
            {"contains": "responsive top nav", "responses": [{"text": "<nav></nav>"}]},
            {"contains": "dedupe(seq)", "responses": [{"kind": "thinking_only"}, {"text": "```python\ndef dedupe(seq):\n    seen=set(); out=[]\n    for x in seq:\n        if x not in seen:\n            seen.add(x); out.append(x)\n    return out\n```"}]},
            {"contains": "parse_csv(text)", "responses": [{"kind": "thinking_only"}, {"text": "not a csv parser"}]},
            {"contains": "debounce(fn, delay)", "responses": [{"kind": "empty"}, {"text": "```javascript\nfunction debounce(fn, delay){let t;return function(...args){clearTimeout(t);t=setTimeout(()=>fn.apply(this,args),delay);};}\n```"}]},
            {"contains": "Four people, A, B, C, and D", "responses": [{"kind": "thinking_only"}, {"kind": "thinking_only"}, {"kind": "thinking_only"}]},
            {"contains": "1000 bottles of wine", "responses": [{"ok": False, "error": "HTTP 500 timeout", "http_status": 500, "error_kind": "harness_error"}, {"text": "binary-coded prisoner testing"}]},
            {"contains": "three closed doors", "responses": [{"ok": False, "error": "HTTP 500 timeout", "http_status": 500, "error_kind": "harness_error"}, {"ok": False, "error": "HTTP 500 timeout", "http_status": 500, "error_kind": "harness_error"}]},
            {"contains": "identical twins", "responses": [{"ok": False, "error": "confirmed capability unavailable", "error_kind": "capability_unavailable"}]},
            {"contains": "wolf, a goat, and a cabbage", "responses": [{"ok": False, "error": "environment limited", "error_kind": "environment_limited"}]},
            {"contains": "Create a 3-step plan", "responses": [{"ok": False, "error": "operator excluded", "error_kind": "operator_excluded"}]},
            {"contains": "Retrieval Augmented Generation", "responses": [{"text": "Retrieval Augmented Generation retrieves relevant documents before generation and uses them as grounding context for a concise answer."}]},
        ]
    }
    script_path = tmp_path / "mock_script.json"
    script_path.write_text(json.dumps(script))
    monkeypatch.setenv("LLM_MODELBENCH_MOCK_SCRIPT", str(script_path))

    cid = "forced_cli_acceptance"
    cli.main([
        "campaign", "run", "--campaign-id", cid, "--mock", "--models", "qwen2.5-coder:14b",
        "--level", "full",
        "--tasks", "json_extract,git_commit,web_nav,py_dedupe,py_csv,js_debounce,reasoning_bridge_crossing,reasoning_poisoned_wine,reasoning_monty_hall,reasoning_birthday_twins,reasoning_wolf_goat_cabbage,agent_plan,wr_rag",
        "--samples", "1", "--num-predict", "256", "--yes", "--live-ui", "off",
        "--allow-host-code-execution", "--unattended-safe",
    ])

    paths = campaign.resolve_paths(cid, campaigns_root=Path("campaigns"))
    manifest = campaign.load_manifest(paths)
    assert [item["state"] for item in manifest.state_history] == [
        "created", "planned", "generating", "recovering", "judging", "packaged"
    ]
    recovery = json.loads(paths.recovery_result.read_text())
    assert {task for action in recovery["actions"] for task in action.get("tasks", [])} == {
        "py_dedupe", "py_csv", "js_debounce", "reasoning_bridge_crossing",
        "reasoning_poisoned_wine", "reasoning_monty_hall",
    }
    rows = [json.loads(line) for line in paths.effective_rows.read_text().splitlines() if line]
    assert Counter(row["result_origin"] for row in rows) == Counter({"primary": 6, "recovered": 4, "recovery_terminal": 2, "judged": 1})
    assert any(row["task"] == "py_csv" and row["result_origin"] == "recovered" and row["effective_score"] == 0 for row in rows)
    assert any(row["terminal_disposition"] == "terminal_thinking_only" for row in rows)
    assert any(row["terminal_disposition"] == "terminal_transient" for row in rows)
    assert json.loads(paths.readiness_json.read_text())["readiness"] == "ready_for_adoption"
    assert campaign.verify_package_details(paths)["valid"] is True
    with zipfile.ZipFile(paths.packages_dir / f"{cid}-review.zip") as archive:
        names = archive.namelist()
    assert not any(name.startswith("evidence/repair_") for name in names)
    assert [name for name in names if name.endswith("/report.html") or name == "reports/report.html"] == ["reports/report.html"]
