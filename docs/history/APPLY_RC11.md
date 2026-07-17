# Apply LLM ModelBench 1.0.0rc11

RC11 is a pre-Rankings-V3 validation release. It adds deterministic watcher replay, a corrected repair dashboard, controlled 64k telemetry validation, operating model cards, recovery-limit semantic cleanup, and a regression freeze.

## Apply

From the current RC10 working tree:

```bash
cd ~/llm-modelbench

git diff > ../llm-modelbench-pre-rc11-local.patch

patch --dry-run -p1 < llm-modelbench-1.0.0rc11-pre-rv3.patch
patch -p1 < llm-modelbench-1.0.0rc11-pre-rv3.patch

python3 -m compileall -q llm_modelbench tests
python3 -m pytest -q
python3 -m llm_modelbench selftest
./llmb --version
```

Expected offline result:

```text
361 passed
SELFTEST: ALL GOOD
llm-modelbench 1.0.0rc11
```

## Offline watcher validation

Direct replay:

```bash
./llmb simulate repair-watch --scenario capability-repair --speed 1.0
./llmb simulate repair-watch --scenario needle-current --speed 1.0
```

Standalone watcher validation:

Terminal 1:

```bash
./llmb-watch --runs-dir runs --follow-queue --layout repair --refresh 0.5 --idle-grace 30
```

Terminal 2:

```bash
./llmb simulate repair-watch \
  --scenario kv-cascade \
  --speed 1.0 \
  --write-only \
  --run-id rc11_watch_fixture
```

No model or GPU work is performed by these fixtures.

## Controlled real-host 64k telemetry validation

This is the only model run requested for RC11. It uses current/default KV and does not mutate systemd or start a q8/q4 cascade.

```bash
./llmb context-profile \
  --model 'deepseek-coder-v2:16b' \
  --run-id rc11_deepseek_coder_v2_16b_64k_profile \
  --runs-dir runs \
  --rankings-out rankings \
  --cards-out model_cards \
  --target-ctx 64000 \
  --gpu-vram-gb 15.93 \
  --emergency-headroom-gb 0.25 \
  --max-spill-gb 2.5 \
  --min-tps 10 \
  --critical-tps 3 \
  --live-ui compact \
  --yes
```

The command exits non-zero when required 64k telemetry is missing. A successful run writes `telemetry_validation.json`, refreshes rankings, and regenerates model cards. The profile row is diagnostic and cannot replace canonical quality evidence.

## Freeze before Rankings V3

After the telemetry profile passes:

```bash
./llmb rankings --runs-dir runs --out rankings --rescan
./llmb model-cards --rankings-dir rankings --runs-dir runs --out model_cards
./llmb freeze \
  --repo-root . \
  --runs-dir runs \
  --rankings-dir rankings \
  --out snapshots/rc11-pre-rv3 \
  --label rc11-pre-rv3
```

Do not begin Rankings V3 until the watcher replay, real-host telemetry validation, model cards, and freeze all pass.
