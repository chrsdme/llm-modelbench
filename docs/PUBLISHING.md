# Publishing and releases

Public repository: `https://github.com/chrsdme/llm-modelbench`

Package and CLI: `llm-modelbench` / `llm_modelbench`

## Release checklist

1. Start from a clean, synchronized branch.
2. Confirm that package metadata, runtime version, README, and the newest changelog entry agree.
3. Run `python tools/release_check.py`, the configured Ruff checks, and `bandit -q -r llm_modelbench tools -x tests -ll`.
4. Run `python -m compileall -q llm_modelbench tests tools`.
5. Run `python -m pytest -q` and `python -m llm_modelbench selftest`.
6. Build both distributions with `python -m build`.
7. Install the wheel into a fresh virtual environment and run the installed CLI self-test.
8. Verify that every packaged task resource loads from the installed wheel.
9. Run release-hygiene and secret scans against the complete Git history.
10. Review generated artifacts and the final diff before tagging.
11. Create and push the release tag only after review. Never force-push a protected release branch.

Public clone instructions use HTTPS. Maintainers may configure authenticated SSH remotes separately.

Release summaries belong in [CHANGELOG.md](../CHANGELOG.md). Historical audit documents belong in [history/](history/), not in the documentation root.
