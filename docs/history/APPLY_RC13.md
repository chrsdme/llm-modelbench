# Apply LLM ModelBench 1.0.0rc13

RC13 adds Rankings V3 operational leaderboards. It does not change task scoring, task definitions, repair policy, or raw evidence selection semantics.

## Apply

```bash
cd ~/llm-modelbench

git diff > ../llm-modelbench-pre-rc13-local.patch

patch --dry-run -p1 \
  < llm-modelbench-1.0.0rc13-rankings-v3.patch

patch -p1 \
  < llm-modelbench-1.0.0rc13-rankings-v3.patch

python3 -m compileall -q llm_modelbench tests
python3 -m pytest -q
python3 -m llm_modelbench selftest
./llmb --version
```

Expected:

```text
373 passed
SELFTEST: ALL GOOD
llm-modelbench 1.0.0rc13
```

## Rebuild Rankings V3

```bash
./llmb rankings --runs-dir runs --out rankings --rescan
```

Expected additional files:

```text
rankings/master_report_v3_data.json
rankings/master_report_v3.html
```

The CLI should print the V3 path:

```text
v3 -> rankings/master_report_v3.html
```

## Verify V3 payload

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path('rankings/master_report_v3_data.json')
data = json.loads(p.read_text())
print(data['schema_version'])
print(data['summary'])
print(sorted(data['use_case_rankings']))
assert data['schema_version'] == 'llm-modelbench.rankings.v3'
assert data['summary']['models'] == 61
assert data['summary']['status_counts'].get('complete') == 61
assert 'coding' in data['use_case_rankings']
assert 'long_context_64k' in data['use_case_rankings']
PY
```

## Regenerate cards and freeze

```bash
./llmb model-cards --rankings-dir rankings --runs-dir runs --out model_cards

./llmb freeze \
  --repo-root . \
  --runs-dir runs \
  --rankings-dir rankings \
  --out snapshots/rc13-rankings-v3 \
  --label rc13-rankings-v3

./llmb freeze --out snapshots/rc13-rankings-v3 --verify
sha256sum -c snapshots/rc13-rankings-v3/SHA256SUMS.txt
snapshots/rc13-rankings-v3/VERIFY.sh
```

RC13 freeze snapshots include both legacy and V3 ranking artifacts.
