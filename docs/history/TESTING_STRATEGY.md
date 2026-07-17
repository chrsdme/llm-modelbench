# Testing strategy for 1.0.0-rc3

## 1. Offline acceptance

```bash
python -m pytest -q
python -m compileall -q llm_modelbench tests
./llmb selftest
./llmb plan --mock --models 'qwen2.5-coder:14b;qwen2.5-vl:7b' --auto --level short
./llmb-run --mock --models 'qwen2.5-coder:14b' --auto --level short \
  --tasks fim_suffix_assertion,agent_native_tool_call --run-id rc3_mock_newlanes --yes --live-ui off
./llmb judge-dumps --everything --runs-dir runs --judge-model mock-judge --mock --dry-run
./llmb rankings --runs-dir runs --out rankings --rescan
```

## 2. Real capability smoke

Use one known model per lane, not the full fleet:

```bash
./llmb plan --models 'qwen2.5-coder:14b;qwen2.5vl:7b;nomic-embed-text:latest' --auto --level short
./llmb-run --models 'qwen2.5-coder:14b' --auto --level short \
  --tasks fim_suffix_assertion,agent_native_tool_call --run-id rc3_native_capabilities --yes --live-ui compact
./llmb-run --models 'qwen2.5vl:7b' --auto --level short \
  --categories ocr,pdf --run-id rc3_vision --yes --live-ui compact
./llmb-run --models 'nomic-embed-text:latest' --auto --level short \
  --categories retrieval --run-id rc3_embedding --yes --live-ui compact
```

Inspect `capability_report.json`, raw rows and reports before a fleet run.

## 3. Post-hoc judge validation

Start with a dry run, then one archived run, then all runs:

```bash
./llmb judge-dumps --everything --runs-dir runs --judge-model qwen2.5:14b --dry-run
./llmb judge-dumps --run-id RUN_ID --judge single --judge-model qwen2.5:14b --yes
./llmb judge-dumps --everything --runs-dir runs --judge single --judge-model qwen2.5:14b --yes
```

Hash one source `raw_results.jsonl` before and after to independently confirm immutability.

## 4. Reliability and fleet rollout

Run the same small representative set five times with identical settings before accepting new ranking precision. Then run a short capability-routed fleet. Use full/needle only for finalists and context-capable candidates. Review tie bands, scope status and model-card history rather than treating decimal differences as decisive.
