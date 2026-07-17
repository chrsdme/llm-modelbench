# LLM ModelBench 1.0.0-rc3 source audit

## Scope

Offline source audit, mock execution and archived-artifact replay. No real model benchmark was required for this audit.

## High-impact findings fixed

1. **Capability override precedence:** explicit profile families were unreachable whenever Ollama returned any capability list. Profiles now take precedence and hybrid lanes are preserved.
2. **Incomplete model interrogation:** native tool calls and suffix/FIM capability were not tested through their actual Ollama API paths. Dedicated probes and benchmark tasks were added.
3. **Vision false-positive risk:** the first probe design exposed the expected token in text. The token now exists only in the image.
4. **Unsafe unattended approval behavior:** non-TTY runs could proceed after merely printing a message. They now stop before model calls unless the exact plan is approved with `--yes`.
5. **No automated retroactive judge:** existing subjective dumps can now be judged singly or across every run through an immutable sidecar.
6. **Ranking evidence loss at presentation layer:** historical rows were stored but not shown in model cards. Cards now include complete attempt history and identify the row selected for current ranking.
7. **Ambiguous ranking precision:** tie bands, scope status, category/class top five and a separate multimodal view are now explicit.
8. **Insufficient model identity reporting:** parameter size, quantization and architecture metadata are carried from `model_identities.json` into model cards.
9. **Profile class/family disagreement:** a class-only name profile could label an embedding-only runtime as reasoning or experimental. Embedding-only runtime evidence now wins over incompatible display-class hints.
10. **Partial-metadata VLM suppression:** InternVL3-8B, Garnet-OCR-7B and VL-1-Coder could still lose vision routing when Ollama returned `completion` without `vision`. Narrow explicit profiles now preserve the fleet routes, and `--auto` probes conservative visual candidates despite partial metadata.
11. **Capability-order inconsistency:** interrogation emitted `text,vision` while classification emitted `vision,text`. Both now use one canonical family order.

## Defensive changes

- report/grade/pack commands now reject a missing run target instead of constructing `Path(None)`;
- capability-specific warm-up avoids chat calls for embedding-only/FIM-only lanes;
- task and model elapsed time are persisted for new rows;
- judge failures are counted separately from successful judgements;
- source signatures include judge and capability sidecars so rankings reimport changed evidence;
- HTML JSON embedding escapes closing script sequences.

## Known limits

- A functional capability probe is evidence, not proof of broad competence; real lane tasks remain authoritative.
- Post-hoc automated judging inherits bias and accuracy limits of the selected local judge.
- Historical rows lacking task hashes cannot prove comparability and are labelled stale/provisional when current scope requires them.
- The current multimodal suite can still produce ceiling ties; harder layout/chart/visual-reasoning fixtures are needed.
- Real provider compatibility for native tool and FIM paths must be smoke-tested against the installed Ollama build and selected models.

## Validation completed

- full pytest suite: `265 passed`;
- Python compileall: passed;
- built-in offline selftest: `SELFTEST: ALL GOOD`;
- focused mock run: native structured tool task and held-out suffix/FIM task both scored 100 and generated reports;
- archived rankings rescan: 104 run directories, 4,302 preserved rows, 67 model identities and 12 multimodal candidates;
- archived post-hoc judge dry-run: 104 runs scanned, 309 eligible subjective rows across 81 runs, 9 rows skipped/already ineligible;
- source raw immutability and sidecar overlay are regression-tested.

The final archived scope classification is 7 complete embedding specialists, 59 provisional models and 1 ineligible model. The master overall table deliberately ranks text-capable assistants only; embedding-only specialists remain in retrieval and embedding-class tables.
