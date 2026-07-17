# Apply RC10

From the current RC9 repository root:

```bash
patch --dry-run -p1 < llm-modelbench-1.0.0rc10-repair-truth.patch
patch -p1 < llm-modelbench-1.0.0rc10-repair-truth.patch
python3 -m compileall -q llm_modelbench tests
python3 -m pytest -q
python3 -m llm_modelbench selftest
./llmb --version
```

Expected:

```text
353 passed
SELFTEST: ALL GOOD
llm-modelbench 1.0.0rc10
```

Then perform only a read-only rankings rescan:

```bash
./llmb rankings --runs-dir runs --out rankings --rescan
```

The existing RC9 InternVL3 FIM repair evidence should be adopted as a measured zero-quality terminal result. No model rerun is required.
