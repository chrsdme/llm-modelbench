# RC21 Stage 2: Multi-GPU Inventory

## Outcome

Stage 2 adds `hardware.detect_gpus() -> list[GPUDevice]` as the primary physical NVIDIA inventory API. `GPUDevice` records NVIDIA physical index, UUID, PCI bus ID, name, total VRAM in MiB, driver version, and compute capability when the driver provides them. The parser preserves `nvidia-smi` row order and represents optional unavailable values as `None`.

The old implementation selected `nvidia-smi` output row zero. `detect_gpu()` remains the temporary RC20 compatibility wrapper and projects the first `detect_gpus()` result into the unchanged `GPUInfo` fields. Its AMD, Apple, and no-device fallbacks are unchanged. Existing live telemetry remains scalar/first-device behavior; per-GPU telemetry is explicitly deferred to Stage 6.

`physical_index` is an NVIDIA host-level identity only. Stage 2 does not infer or persist CUDA runtime-visible indices, `CUDA_VISIBLE_DEVICES`, runtime selection, backend selection, device splitting, or server configuration. Those mappings belong to the later runtime-profile work.

Doctor now includes an additive `gpus` JSON list and per-device text lines while retaining its existing scalar `GPU:` line.

## Files changed

- `llm_modelbench/hardware.py`
- `llm_modelbench/doctor.py`
- `tests/test_hardware_inventory.py`
- `docs/rc21/RC21_PROGRESS.md`
- `docs/rc21/stage-02-multigpu-inventory.md`

## Tests

Focused validation passed:

```text
python3 -m pytest -q tests/test_hardware_inventory.py tests/test_config_validation.py tests/test_planner_doctor_grade.py tests/test_repair.py
59 passed
```

The focused inventory tests cover multiple rows, source order, UUID/PCI identity, unavailable optional values, compute-capability-query fallback, `detect_gpu()` first-device projection, no-GPU behavior, and doctor scalar/inventory rendering.

Full validation passed:

```text
python3 -m compileall -q llm_modelbench tests
python3 -m pytest -q
538 passed
./llmb selftest
SELFTEST: ALL GOOD
python3 tools/release_check.py
RELEASE CHECK: PASS
git diff --check
passed
```

## Real-hardware acceptance

The Codex execution environment could not communicate with the NVIDIA driver and returned an empty inventory. Final acceptance was therefore performed separately on the real AI-PC host using the new read-only API.

The real host detected two ordered physical NVIDIA devices:

| Physical index | Device | UUID | PCI bus ID | VRAM | Compute capability |
| ---: | --- | --- | --- | ---: | ---: |
| 0 | NVIDIA GeForce RTX 5060 Ti | `GPU-78077308-f4b2-3330-6d4e-19581d7b1511` | `00000000:01:00.0` | 16311 MiB | 12.0 |
| 1 | NVIDIA GeForce RTX 3060 | `GPU-5b99bce2-35ab-f6db-857b-72162069fa72` | `00000000:05:00.0` | 12288 MiB | 8.6 |

Both devices have distinct UUID and PCI identities, the expected stable ordering, correct reported VRAM totals, and correct compute capabilities. Stage 2 real-hardware acceptance therefore passed.

This was an inventory-only validation. It did not benchmark, stress, start, stop, or reconfigure either GPU or any inference service.

## Known limitations and rollback

- The inventory is NVIDIA-only; existing scalar AMD/Apple fallbacks are retained but do not yet return `GPUDevice` entries.
- A legacy NVIDIA driver that rejects `compute_cap` is retried without it and reports `compute_capability: None`.
- Current telemetry and downstream run evidence remain scalar until Stage 6.
- Rollback is limited to removing `detect_gpus()` and doctor inventory display; `detect_gpu()` and `GPUInfo` retain their prior contract.

## Proposed Stage 3 objective

Define a capability-oriented backend protocol and preserve the current Ollama/Mock adapters without behavior drift. Do not add llama-server implementation or runtime-profile selection in Stage 3.
