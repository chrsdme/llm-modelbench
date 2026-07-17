# LLM ModelBench 1.0.0rc13 Rankings V3 audit

## Purpose

RC13 adds an operational Rankings V3 layer over the already-correct RC12 evidence model. It is a presentation and derived-ranking layer only. It does not rewrite raw rows, task scores, scorer logic, task hashes, or repair outcomes.

## New artifacts

Every rankings rebuild now writes:

- `rankings/master_report_v3_data.json`
- `rankings/master_report_v3.html`

Legacy files remain:

- `rankings/master_raw.jsonl`
- `rankings/master_summary.json`
- `rankings/master_report_data.json`
- `rankings/master_report.html`

## Use-case rankings

V3 generates operational leaderboards for:

- overall text assistants;
- coding;
- reasoning;
- agentic/tool use;
- 64k context readiness;
- multimodal/OCR/PDF;
- retrieval/RAG;
- embedding specialists;
- small-fast models.

The V3 score is derived from current quality, status confidence, coverage, speed, size efficiency, and long-context readiness where relevant.

## Evidence labels

V3 preserves terminal evidence labels:

- `capability_limited`;
- `capability_measured_failure`;
- `recovery_limited`;
- target context status;
- error and warning states.

Unavailable capabilities remain visible as exclusions. Measured capability failures remain visible as zero-quality outcomes. Recovery-limited tasks remain terminal evidence, not missing work.

## Long-context policy

A 64k context ranking entry requires current long-context evidence. V3 distinguishes:

- technically verified context;
- target-context decode speed;
- slow or impractical speed;
- behavior warnings;
- not-profiled or not-verified models.

Needle success alone is not represented as proof of long-horizon agentic reliability.

## UI behavior

The V3 HTML includes:

- dark high-contrast layout;
- model search;
- status filter;
- 64k status filter;
- minimum-quality slider;
- use-case tabs;
- operational badges;
- table and card views.

## Validation

Sandbox validation:

```text
compileall passed
pytest 373 passed
SELFTEST: ALL GOOD
version 1.0.0rc13
node --check generated V3 script passed
V3 generated successfully from the reviewed 61-model ranking payload
```

No live Ollama call, model inference, GPU allocation, sudo command, service restart, or systemd mutation was performed while building RC13.
