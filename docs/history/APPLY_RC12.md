# Apply LLM ModelBench 1.0.0rc12

From the repository root:

```bash
git diff > ../llm-modelbench-pre-rc12-local.patch
patch --dry-run -p1 < llm-modelbench-1.0.0rc12-pre-rv3-truth.patch
patch -p1 < llm-modelbench-1.0.0rc12-pre-rv3-truth.patch
python3 -m compileall -q llm_modelbench tests
python3 -m pytest -q
python3 -m llm_modelbench selftest
./llmb --version
```

Expected: `370 passed`, `SELFTEST: ALL GOOD`, `llm-modelbench 1.0.0rc12`.

Use a new run ID for the controlled 64k profile. RC12 refuses to overwrite or append to a non-empty diagnostic run directory.
