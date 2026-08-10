# H4 - Release Hygiene, Wheel Smoke, SHA Errata

## Scope
- Branch: `rc21-post1-topology-budget`
- Baseline commit: `41d54cf2def2b868284be2bc3cd7e65d578646f9`
- Stage: H4 only

## Requirements Implemented
- `tools/release_check.py` now always scans tracked files, regardless of directory name.
- Untracked local operational/audit artifacts under local-only directories remain ignored.
- Regression tests prove a tracked `codex_logs/test.md` private-home spill fails release checking, while an untracked local operational log under `codex_logs` is ignored.
- Tracked stage logs are treated as release content and are scanned by release hygiene.
- Created `codex_logs/Final Hardening/commit_sha_errata.md` with Git-resolved canonical SHAs and parent/subject map.
- Historical audit `.log` files were not rewritten.

## Wheel Smoke
- Isolated build command: `./.venv/bin/python -m build`.
- Build result: succeeded with isolated `venv+pip`; no `--no-isolation` fallback was used.
- Installed wheel into temporary venv with `--force-reinstall --no-index --no-deps`.
- Smoke was run from outside the repository with `PYTHONPATH` unset.
- Import path: `/tmp/llmb-wheel-smoke-TrLeSv/lib/python3.14/site-packages/llm_modelbench/__init__.py`.
- `python -m llm_modelbench --help` -> passed.
- `python -m llm_modelbench selftest` -> `SELFTEST: ALL GOOD`.

## Focused Validation
- `./.venv/bin/pytest -q tests/test_release_hygiene.py tests/test_docs_hygiene.py` -> 11 passed.
- `./.venv/bin/ruff check tools/release_check.py tests/test_release_hygiene.py tests/test_docs_hygiene.py` -> passed.
- `python -m compileall -q tools/release_check.py tests/test_release_hygiene.py tests/test_docs_hygiene.py` -> passed.
- `git diff --check` -> passed.

## Manual Inspection
- Confirmed `repository_files()` unions all tracked files with filtered untracked files.
- Confirmed tracked `codex_log/`, `codex_logs/`, and `codex_prompts/` files are no longer hidden by directory name.
- Confirmed untracked local operational logs can remain ignored.
- Confirmed canonical SHA table was generated from Git, including Stage 3B and Stage 4B corrected full SHAs.
- Confirmed the wheel smoke imported from the temporary venv, not the checkout.
- Confirmed no real model, judge, recovery, catch-up, adoption, evidence rewrite, or push work occurred.
