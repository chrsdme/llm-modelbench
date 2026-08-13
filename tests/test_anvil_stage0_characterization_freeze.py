"""Anvil Stage 0.0 — behavioral characterization freeze.

This module is the regression baseline the Anvil master plan
(local_only/anvil/ANVIL_MASTER_PLAN.md, Stage 0.0) requires before any
other Anvil stage touches evidence storage, capability routing, the CLI
surface, or campaign logic: it proves current behavior is understood and
frozen, so every later stage's tests must show *intentional* divergence
from this baseline (a deliberately updated fixture, with a reason) rather
than silent drift.

Everything here runs fully offline against the deterministic --mock
client. No GPU, no live Ollama/llama.cpp endpoint, no model download.

Fixtures live under tests/fixtures/anvil_stage0_baseline/. Regenerate them
(after an *intentional* behavior change) by running this module's
`_generate_fixtures()` directly:
    python -c "from tests.test_anvil_stage0_characterization_freeze import _generate_fixtures as g; g()"
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "anvil_stage0_baseline"
CLI_HELP_DIR = FIXTURE_ROOT / "cli_help"
MOCK_RUN_DIR = FIXTURE_ROOT / "mock_run"
RANKINGS_DIR = FIXTURE_ROOT / "rankings_snapshot"

_PATH_LIKE = re.compile(r"^(/tmp/|/home/|/private/tmp/|/private/var/)\S*$")


def _normalize(value: Any) -> Any:
    """Replace known-volatile fields (timestamps, absolute run-dir paths,
    and capability-probe wall-clock timing) with stable sentinels so frozen
    fixtures are portable across machines and re-runs. Applied identically
    at fixture-generation time and at comparison time — see module
    docstring.

    `evidence_hash`/`capability_evidence_hash` are included here as a
    documented finding, not an oversight: `capabilities.interrogate_model`
    (capabilities.py) computes this hash over the *entire* probe payload,
    which includes `probes.*.elapsed_seconds` (real wall-clock probe
    latency) — so the hash is not actually a stable content-identity hash,
    it changes on every call even with byte-identical input/output. Proven
    directly: two consecutive in-process calls to `interrogate_model` on
    the same mock model produced two different `evidence_hash` values,
    differing only in `elapsed_seconds`. Worth carrying into Anvil Stage
    2's `CapabilityObservation.evidence_hash` design: a stable identity
    hash must exclude timing fields, which this one does not.
    """
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, val in value.items():
            if key == "timestamp":
                out[key] = "<TIMESTAMP>"
            elif key in ("created_from", "created_at", "generated_at", "updated_at"):
                out[key] = "<GENERATED_AT>"
            elif key in ("evidence_hash", "capability_evidence_hash"):
                out[key] = "<VOLATILE_TIMING_HASH>"
            elif key == "elapsed_seconds":
                out[key] = "<ELAPSED>"
            elif key in ("rt_tag", "import_tag"):
                # rankings.new_import_tag() generates a fresh random short
                # ID per import by design (rankings.py) -- expected to vary.
                out[key] = "<IMPORT_TAG>"
            elif key in (
                "task_wall_seconds",
                "model_elapsed_seconds",
                "wall_seconds",
                "total_wall_seconds",
            ):
                # runner.py measures task_wall_seconds/model_elapsed_seconds
                # with time.perf_counter() around real (if fast) harness
                # execution even under --mock -- genuine wall-clock timing,
                # not a canned mock value. rankings.py's _task_seconds()
                # sums task_wall_seconds into wall_seconds/total_wall_seconds
                # per category/model, so the variance propagates. All four
                # vary run to run by design, not by bug.
                out[key] = "<WALL_CLOCK>"
            elif key in ("power_mean_w", "vram_peak_mb", "temp_peak_c"):
                # hardware.py's Telemetry.stop() (~line 213) aggregates
                # samples collected by a background thread polling on a
                # real time.sleep() interval -- sample count (and so the
                # aggregate) depends on real task wall-clock duration, even
                # under --mock. Not a canned constant like tok/s or
                # ttft_ms; expected to vary run to run.
                out[key] = "<TELEMETRY>"
            elif key in ("identity_hash", "runtime_identity_hash", "runtime_variant_id"):
                # Real finding, caught by test_docs_hygiene.py's own
                # known-host-hardware-identifier check on first fixture
                # generation attempt: these hashes are derived from real
                # physical GPU UUIDs/PCI bus IDs on whatever host generates
                # them (see the GPU-UUID/PCI-bus-ID scrub below) -- so they
                # are host-identifying, not just run-to-run volatile.
                # Portable fixtures must not bake in a contributor's actual
                # hardware identity.
                out[key] = "<HOST_DERIVED_HASH>"
            elif key in (
                "physical_gpu_uuids", "declared_device_order", "gpu_uuid_assignment",
            ):
                # Real finding, caught the same way as identity_hash above,
                # by an actual CI run rather than by inspection: real GPU
                # topology detection is NOT mocked by --mock, so these lists'
                # *length* (not just their UUID values, already scrubbed by
                # the GPU-UUID regex below) reflects whatever host generated
                # the fixture -- 2 GPUs on the machine that froze this
                # fixture, 0 GPUs on a standard GitHub Actions runner. A
                # value-level scrub can turn a real UUID into <GPU_UUID_0>,
                # but it can't turn a 2-element list into an empty one --
                # collapse the whole field to one topology-agnostic
                # placeholder rather than attempt to preserve shape.
                out[key] = "<GPU_TOPOLOGY_LIST>"
            elif key == "pci_bus_ids":
                # Same finding, dict form (GPU UUID -> PCI bus ID).
                out[key] = "<GPU_TOPOLOGY_MAP>"
            else:
                out[key] = _normalize(val)
        return out
    if isinstance(value, list):
        items = [_normalize(item) for item in value]
        # Finding: rankings.py's per-model `history` list is built with
        # order that varies across separate process invocations even given
        # byte-identical input (confirmed: sorting by `task` and comparing
        # field-by-field shows zero remaining differences) -- almost
        # certainly PYTHONHASHSEED-driven dict/set iteration order
        # somewhere in the history-building code (CPython randomizes hash
        # seed per process by default). Canonicalize order for comparison
        # purposes here rather than chase the exact line; this is the
        # correct fix regardless of root cause, since positionally
        # comparing an unordered collection is wrong either way. Worth a
        # closer look whenever Stage 6B rebuilds this data model.
        if items and all(isinstance(i, dict) and "task" in i for i in items):
            items = sorted(items, key=lambda i: (str(i.get("task")), str(i.get("category"))))
        return items
    if isinstance(value, str) and _PATH_LIKE.match(value):
        return "<RUN_DIR>"
    return value


# Real finding (not anticipated when this module was first written): GPU
# UUIDs and PCI bus IDs from the *actual host running fixture generation*
# leak into runtime-identity evidence even under --mock, because hardware
# topology detection reflects real host capacity regardless of mock mode --
# only the model-inference client is faked. Caught by this repo's own
# test_docs_hygiene.py::test_known_host_hardware_identifiers_are_not_publicly_tracked
# on the first fixture-generation attempt (it already guards against exactly
# this class of leak, precedent from the RC15 public-readiness audit). These
# appear as both dict *values* and dict *keys* (e.g. `pci_bus_ids: {"GPU-
# xxx...": "0000:05:00.0"}`), so a value-only walk over the parsed structure
# can't catch the key form -- scrub the serialized text directly instead,
# with a stable per-fixture positional remap so structure (how many GPUs,
# whether a UUID recurs across fields) stays checkable without the real
# identifiers.
_GPU_UUID_RE = re.compile(r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_PCI_BUS_ID_RE = re.compile(r"\b[0-9a-fA-F]{8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F]\b")


def _scrub_host_hardware_identifiers(text: str) -> str:
    gpu_uuids = sorted(set(_GPU_UUID_RE.findall(text)))
    for index, uuid in enumerate(gpu_uuids):
        text = text.replace(uuid, f"<GPU_UUID_{index}>")
    pci_ids = sorted(set(_PCI_BUS_ID_RE.findall(text)))
    for index, pci_id in enumerate(pci_ids):
        text = text.replace(pci_id, f"<PCI_BUS_ID_{index}>")
    return text


def _normalize_jsonl_text(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    normalized = [json.dumps(_normalize(json.loads(line)), sort_keys=True) for line in lines]
    return _scrub_host_hardware_identifiers("\n".join(normalized) + "\n")


def _normalize_json_text(text: str) -> str:
    return _scrub_host_hardware_identifiers(
        json.dumps(_normalize(json.loads(text)), sort_keys=True, indent=2) + "\n"
    )


_CSV_HOST_DERIVED_HASH_COLUMNS = ("runtime_identity_hashes",)


def _normalize_csv_text(text: str) -> str:
    """scorecard.csv's runtime_identity_hashes column embeds a
    RuntimeIdentity.identity_hash value per model -- like the JSON
    identity_hash/runtime_identity_hash/runtime_variant_id fields
    _normalize() already handles, this is derived from real physical GPU
    UUIDs/PCI bus IDs on whatever host generated it, so it is host-
    identifying, not run-to-run stable content. A raw byte comparison
    across two different hosts (e.g. the machine that froze this fixture
    vs. an ephemeral CI runner) never matches on this column even though
    every other field in the row is genuinely deterministic under --mock.
    Confirmed root cause of this project's CI always failing on
    test_mock_run_evidence_matches_frozen_baseline: scorecard.csv was
    copied and compared raw (no normalization at all), unlike every other
    frozen evidence file, which does go through _normalize() for exactly
    this concern."""
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return text
    header = rows[0]
    hash_columns = [
        index for index, column in enumerate(header) if column in _CSV_HOST_DERIVED_HASH_COLUMNS
    ]
    for row in rows[1:]:
        for index in hash_columns:
            if index >= len(row) or not row[index]:
                continue
            try:
                hashes = json.loads(row[index])
            except json.JSONDecodeError:
                continue
            row[index] = json.dumps(["<HOST_DERIVED_HASH>" for _ in hashes])
    output = io.StringIO()
    csv.writer(output, lineterminator="\n").writerows(rows)
    return output.getvalue()


def _run_mock_benchmark(out_dir: Path, run_id: str) -> Path:
    """Run a deterministic, fully offline mock benchmark. Returns the run directory."""
    env = dict(os.environ)
    env["PYTHON_COLORS"] = "0"
    cmd = [
        sys.executable,
        "-m",
        "llm_modelbench",
        "run",
        "--mock",
        "--level",
        "short",
        "--models",
        "qwen2.5-coder:14b;llama3.1:8b",
        "--auto",
        "--allow-host-code-execution",
        "--run-id",
        run_id,
        "--out",
        str(out_dir),
        "--judge",
        "single",
        "--judge-model",
        "nomic-embed-text:latest",
        "--live-ui",
        "off",
        "--no-ranking-update",
        "--yes",
    ]
    result = subprocess.run(
        cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, (
        f"mock benchmark run failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return out_dir / run_id


# Files whose content is verified deterministic (proven by running the mock
# benchmark twice and diffing) and worth an exact frozen comparison. Other
# files in the evidence tree are checked for presence only (see
# EXPECTED_EVIDENCE_FILES) — freezing their exact bytes wasn't warranted by
# this pass (either clearly derived/redundant with what's frozen here, or
# not yet confirmed stable across environments).
FROZEN_TEXT_FILES = ["routing.md", "prune.md", "clones.md"]
FROZEN_CSV_FILES = ["scorecard.csv"]
FROZEN_JSON_FILES = ["summary.json"]
FROZEN_JSONL_FILES = ["raw_results.jsonl"]

EXPECTED_EVIDENCE_FILES = {
    "capability_report.json",
    "clones.md",
    "config.json",
    "filters.json",
    "fingerprints.json",
    "model_identities.json",
    "prune.md",
    "ranking_scope.json",
    "raw_results.jsonl",
    "regression.md",
    "report.html",
    "routing.md",
    "run_validity.json",
    "runtime_identity.json",
    "scorecard.csv",
    "scorecard.md",
    "skipped_models.json",
    "status.json",
    "summary.json",
    "summary_meta.json",
}


def _generate_fixtures() -> None:
    """One-off generator. Not run by pytest — invoked manually to (re)create
    the frozen fixtures after a reviewed, intentional behavior change."""
    import shutil
    import tempfile

    CLI_HELP_DIR.mkdir(parents=True, exist_ok=True)
    from llm_modelbench.cli import build_parser

    def walk(parser: argparse.ArgumentParser, path: list[str]) -> None:
        name = "top" if not path else "_".join(path)
        (CLI_HELP_DIR / f"{name}.txt").write_text(_normalize_argparse_help(parser.format_help()))
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                for choice, subparser in action.choices.items():
                    walk(subparser, path + [choice])

    os.environ["PYTHON_COLORS"] = "0"
    walk(build_parser(), [])

    with tempfile.TemporaryDirectory(prefix="anvil-stage0-gen-") as tmp:
        tmp_path = Path(tmp)
        run_dir = _run_mock_benchmark(tmp_path / "runs", "anvil_stage0_baseline")

        MOCK_RUN_DIR.mkdir(parents=True, exist_ok=True)
        present = {p.name for p in run_dir.iterdir() if p.is_file()}
        (MOCK_RUN_DIR / "_file_manifest.json").write_text(
            json.dumps(sorted(present), indent=2) + "\n"
        )
        for name in FROZEN_TEXT_FILES:
            shutil.copy(run_dir / name, MOCK_RUN_DIR / name)
        for name in FROZEN_CSV_FILES:
            text = (run_dir / name).read_text()
            (MOCK_RUN_DIR / name).write_text(_normalize_csv_text(text))
        for name in FROZEN_JSON_FILES:
            text = (run_dir / name).read_text()
            (MOCK_RUN_DIR / name).write_text(_normalize_json_text(text))
        for name in FROZEN_JSONL_FILES:
            text = (run_dir / name).read_text()
            (MOCK_RUN_DIR / name).write_text(_normalize_jsonl_text(text))

        rankings_out = tmp_path / "rankings"
        env = dict(os.environ)
        env["PYTHON_COLORS"] = "0"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "llm_modelbench",
                "rankings",
                "--runs-dir",
                str(tmp_path / "runs"),
                "--out",
                str(rankings_out),
                "--rescan",
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        RANKINGS_DIR.mkdir(parents=True, exist_ok=True)
        summary_text = (rankings_out / "master_summary.json").read_text()
        (RANKINGS_DIR / "master_summary.json").write_text(_normalize_json_text(summary_text))

    print("Fixtures (re)generated under", FIXTURE_ROOT)


# --------------------------------------------------------------------------
# Regression tests
# --------------------------------------------------------------------------


def _all_help_fixture_names() -> list[str]:
    return sorted(p.stem for p in CLI_HELP_DIR.glob("*.txt"))


def _normalize_argparse_help(text: str) -> str:
    """argparse renamed its default options-group heading from "optional
    arguments:" (Python <=3.9) to "options:" (Python >=3.10) -- a cosmetic
    change internal to argparse itself, not a change to this project's CLI
    surface. Fixtures are frozen under one Python version; without this
    normalization, every --help comparison fails on whichever Python major
    version wasn't used to generate the fixture (confirmed: this broke
    every test_cli_help_matches_frozen_baseline case on the CI matrix's
    Python 3.9 leg, every run, since the fixtures were frozen under
    3.10+)."""
    return text.replace("optional arguments:", "options:")


def _collapse_whitespace(text: str) -> str:
    """argparse's HelpFormatter can wrap the same content at different
    column positions across Python versions/patch releases -- confirmed
    directly: a subparsers choice list's trailing "..." landed on its own
    wrapped line under Python 3.9 but stayed on the closing-brace line
    under the version that froze this fixture, with byte-identical content
    otherwise. Used only at comparison time (not when writing the frozen
    .txt fixtures, which stay human-readable multi-line on disk) so the
    test tolerates *where* argparse chose to wrap a line while still
    catching a real surface change -- an added/removed/renamed flag or
    changed help text still changes the collapsed text, only the wrap
    position is ignored."""
    return re.sub(r"\s+", " ", text).strip()


@pytest.mark.parametrize("name", _all_help_fixture_names())
def test_cli_help_matches_frozen_baseline(name: str) -> None:
    """Every command/sub-subcommand's --help text matches the Stage 0.0
    baseline. A deliberate CLI surface change must update the fixture in
    the same commit, not drift silently."""
    from llm_modelbench.cli import build_parser

    os.environ["PYTHON_COLORS"] = "0"
    parser = build_parser()
    path = [] if name == "top" else name.split("_")
    node = parser
    if path:
        for part in path:
            sub_action = next(
                a for a in node._actions if isinstance(a, argparse._SubParsersAction)
            )
            node = sub_action.choices[part]
    expected = (CLI_HELP_DIR / f"{name}.txt").read_text()
    actual = _normalize_argparse_help(node.format_help())
    assert _collapse_whitespace(actual) == _collapse_whitespace(expected)


def test_cli_help_fixture_set_is_complete() -> None:
    """Catches added/removed commands: the fixture directory must cover
    every command currently exposed by build_parser(), no more, no less."""
    from llm_modelbench.cli import build_parser

    discovered: set[str] = set()

    def walk(parser: argparse.ArgumentParser, path: list[str]) -> None:
        discovered.add("top" if not path else "_".join(path))
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                for choice, subparser in action.choices.items():
                    walk(subparser, path + [choice])

    walk(build_parser(), [])
    frozen = set(_all_help_fixture_names())
    assert discovered == frozen, (
        f"CLI surface changed since the Stage 0.0 freeze.\n"
        f"Added commands: {sorted(discovered - frozen)}\n"
        f"Removed commands: {sorted(frozen - discovered)}\n"
        f"If intentional, regenerate fixtures (see module docstring)."
    )


def test_mock_run_evidence_matches_frozen_baseline(tmp_path: Path) -> None:
    """A fresh deterministic --mock run must reproduce the frozen Stage 0.0
    baseline exactly (after normalizing known-volatile fields: timestamps,
    absolute run-directory paths)."""
    run_dir = _run_mock_benchmark(tmp_path / "runs", "anvil_stage0_baseline")

    present = {p.name for p in run_dir.iterdir() if p.is_file()}
    expected_manifest = set(
        json.loads((MOCK_RUN_DIR / "_file_manifest.json").read_text())
    )
    assert present == expected_manifest, (
        f"Evidence tree file set changed since the Stage 0.0 freeze.\n"
        f"Added: {sorted(present - expected_manifest)}\n"
        f"Removed: {sorted(expected_manifest - present)}"
    )
    assert present >= EXPECTED_EVIDENCE_FILES

    for name in FROZEN_TEXT_FILES:
        actual = (run_dir / name).read_text()
        expected = (MOCK_RUN_DIR / name).read_text()
        assert actual == expected, f"{name} diverged from the Stage 0.0 baseline"

    for name in FROZEN_CSV_FILES:
        actual = _normalize_csv_text((run_dir / name).read_text())
        expected = (MOCK_RUN_DIR / name).read_text()
        assert actual == expected, f"{name} diverged from the Stage 0.0 baseline"

    for name in FROZEN_JSON_FILES:
        actual = _normalize_json_text((run_dir / name).read_text())
        expected = (MOCK_RUN_DIR / name).read_text()
        assert actual == expected, f"{name} diverged from the Stage 0.0 baseline"

    for name in FROZEN_JSONL_FILES:
        actual = _normalize_jsonl_text((run_dir / name).read_text())
        expected = (MOCK_RUN_DIR / name).read_text()
        assert actual == expected, f"{name} diverged from the Stage 0.0 baseline"


def test_rankings_formula_matches_frozen_baseline(tmp_path: Path) -> None:
    """The rankings aggregation formula, run against the same deterministic
    mock evidence used throughout this module (a fixed synthetic input in
    every sense that matters — MockClient returns canned, reproducible
    responses), must reproduce the frozen per-model summary exactly."""
    runs_dir = tmp_path / "runs"
    _run_mock_benchmark(runs_dir, "anvil_stage0_baseline")

    rankings_out = tmp_path / "rankings"
    env = dict(os.environ)
    env["PYTHON_COLORS"] = "0"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "llm_modelbench",
            "rankings",
            "--runs-dir",
            str(runs_dir),
            "--out",
            str(rankings_out),
            "--rescan",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr

    actual = _normalize_json_text((rankings_out / "master_summary.json").read_text())
    expected = (RANKINGS_DIR / "master_summary.json").read_text()
    assert actual == expected, "rankings formula output diverged from the Stage 0.0 baseline"


if __name__ == "__main__":
    _generate_fixtures()
