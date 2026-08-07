# RC21 Master Plan

## Product contract

RC21 adds external `llama-server` without demoting Ollama. It discovers healthy local configured/running Ollama and llama-server environments, selects the only viable one automatically, and otherwise asks the operator. Unattended execution fails before model work when selection is ambiguous. A selected runtime profile is reusable and freezes backend, endpoint, server/runtime, physical and runtime-visible GPU identity, and relevant flags into campaign, row, and report evidence.

For a single GPU, recommend Ollama unless the operator explicitly selects llama.cpp/custom flags. For multiple GPUs, recommend llama.cpp when a healthy llama-server profile exists because explicit layer split is available. Recommendation is never silent selection when more than one viable environment exists.

External llama-server remains external: RC21 never launches, restarts, stops, or reconfigures it. A profile normally serves one already-loaded model. One-model llama.cpp runs/campaigns are supported per endpoint/profile; multiple endpoints are separate profiles; inventory reports served models rather than GGUF files; a multi-model campaign fails closed unless that backend genuinely supports switching. Post-hoc judging must use a separate compatible judge runtime profile or preserve the existing external-judge blocker.

Canonical scores, ranking rules, and evidence semantics remain unchanged. New hardware, telemetry, and fit fields are operational evidence, not scoring inputs.

## Stage 0: Working-tree preparation outside Codex

- **Objective:** Human reviews/stashes unrelated work and provides the intended branch/worktree state.
- **Affected files:** none by RC21 code.
- **Non-goals:** no cleanup, reset, commit, service, model, or environment action by the implementation stages.
- **Compatibility constraints:** preserve user changes and historical evidence.
- **Tests:** `git status --short`, `git diff --check` after human preparation.
- **Real-hardware acceptance:** none.
- **Rollback boundary:** human-owned working-tree action only.
- **Documentation outputs:** update `RC21_PROGRESS.md` with human approval.
- **Exact next action:** human approves a clean reviewed tree and Stage 2.

## Stage 1: Architecture audit and durable plan

- **Objective:** establish the source-derived RC21 boundary and migration sequence.
- **Affected files:** `AGENTS.md`, `docs/rc21/*`, `docs/prompts/rc21/*` only.
- **Non-goals:** production/test changes, service interaction, models, rankings, version, release heading.
- **Compatibility constraints:** documentation must not contain personal absolute paths or generated evidence.
- **Tests:** compileall, pytest, selftest, release check, diff check.
- **Real-hardware acceptance:** none; record rather than fabricate evidence.
- **Rollback boundary:** delete only Stage 1 documentation under review.
- **Documentation outputs:** this plan, source audit, progress record, deferred scope, cleaned prompt.
- **Exact next action:** begin Stage 2 only after human review.

## Stage 2: Arbitrary-N real GPU inventory with compatibility wrapper

- **Objective:** introduce an ordered inventory of every real GPU returned by supported system tools, retaining `detect_gpu()`/`GPUInfo` scalar compatibility behavior until consumers migrate.
- **Affected files:** `hardware.py`, `config.py`, `doctor.py`, `runner.py`, `repair.py`, `context_profile.py`, `progress.py`, `watch.py`, `inline_ui.py`, `interactive_nav.py`, and focused tests.
- **Non-goals:** no runtime selection, backend client, explicit layer split, or telemetry schema redesign.
- **Compatibility constraints:** preserve existing `GPUInfo` fields and `detect_gpu()` result for old callers; assign physical index, UUID and PCI bus ID where available, and retain unavailable values explicitly. Keep physical index distinct from `CUDA_VISIBLE_DEVICES`/runtime-visible index.
- **Tests:** parser fixtures for zero/one/many NVIDIA devices, partial UUID/PCI data, AMD/Apple fallback, wrapper behavior, config budget compatibility, doctor/watch rendering.
- **Real-hardware acceptance:** verify all visible physical devices on an NVIDIA multi-GPU host and a constrained-visible-device process; capture durable summary only outside the repository.
- **Rollback boundary:** new inventory API can be removed while scalar wrapper stays intact.
- **Documentation outputs:** inventory identity contract and compatibility note.
- **Exact next action:** Stage 3 defines the backend protocol against this inventory API.

## Stage 3: Backend protocol and Ollama adapter preservation

- **Objective:** define the minimal capability-oriented backend protocol and adapt existing Ollama/Mock implementations without behavior drift.
- **Affected files:** `ollama.py`, new backend/protocol module, `cli.py`, `planner.py`, `runner.py`, `capabilities.py`, `judge.py`, `judge_dumps.py`, `repair.py`, and tests.
- **Non-goals:** no llama-server HTTP implementation yet, no profile persistence, and no scoring changes.
- **Compatibility constraints:** preserve current Ollama endpoint/config/env handling and all existing client result shapes. Split optional features (tools, suffix/FIM, embeddings, loaded-model stats, unload/flush, service repair) from basic generation.
- **Tests:** contract tests run against `MockClient` and `OllamaClient` fixtures; existing API regression tests; explicit unsupported-feature results rather than fallback guesses.
- **Real-hardware acceptance:** confirm an Ollama smoke plan/run maintains identity, task routing, and report output.
- **Rollback boundary:** CLI defaults to the preserved Ollama adapter.
- **Documentation outputs:** protocol and capability matrix.
- **Exact next action:** Stage 4 persists and selects runtime profiles using the protocol.

## Stage 4: Runtime profile schema plus local discovery/selection

- **Objective:** discover healthy local configured/running Ollama and llama-server candidates, make selection deterministic/interactive as required, and save reusable profiles.
- **Affected files:** `config.py`, `cli.py`, new runtime-profile/discovery module, `doctor.py`, `planner.py`, docs, examples, tests.
- **Non-goals:** no daemon/service management; no automatic model launch/switch; no endpoint mutation.
- **Compatibility constraints:** migrate `ollama_url` and `LLM_MODELBENCH_OLLAMA_URL` into an Ollama default/profile without breaking them. Profile identity includes backend, endpoint, discovered server version/capabilities, custom flags, selected physical GPU IDs, runtime-visible mapping, and profile digest. Exactly one viable profile auto-selects; zero reports diagnostics; multiple require operator selection; unattended multiple viable profiles fail before model work.
- **Tests:** discovery fixtures for zero/one/multiple healthy/unhealthy endpoints, selection precedence, noninteractive ambiguity failure, profile round trip/digest, recommendation policy.
- **Real-hardware acceptance:** verify a local Ollama profile and an external llama-server profile independently without lifecycle operations.
- **Rollback boundary:** legacy Ollama URL remains usable as an implicit migrated profile.
- **Documentation outputs:** profile format, discovery security/privacy rules, operator selection examples.
- **Exact next action:** Stage 5 implements the external profile backend.

## Stage 5: External llama-server backend

- **Objective:** make an already-running external llama-server first-class for supported generation/inventory/capability operations.
- **Affected files:** new llama-server adapter, backend protocol modules, `cli.py`, `planner.py`, `runner.py`, `capabilities.py`, `doctor.py`, `judge.py`, `campaign.py`, tests, docs.
- **Non-goals:** managed launch/restart/stop/reconfiguration, scanning GGUF files, automatic fleet switching, pretend Ollama service/KV support.
- **Compatibility constraints:** model inventory describes endpoint-served models only. A one-model endpoint supports one-model runs/campaigns. Multi-model campaigns must fail closed unless proven switching capability exists. Ollama `OllamaServiceController`, KV cascade, loaded model semantics, and Modelfile assumptions remain Ollama-only; llama.cpp reports unsupported operation codes. Judge selection requires a distinct eligible profile or retains `external_judge` blocker.
- **Tests:** OpenAI-compatible/llama-server response fixtures, one-model acceptance, unsupported tools/FIM/embeddings behavior, no-switch multi-model failure, separate judge profile/blocker paths, Ollama regression suite.
- **Real-hardware acceptance:** query a pre-existing llama-server, execute one approved one-model smoke run, and prove no lifecycle command was attempted.
- **Rollback boundary:** disable the llama-server adapter/profile type; Ollama path remains unchanged.
- **Documentation outputs:** endpoint requirements, one-model limitation, separate-endpoint and judge guidance.
- **Exact next action:** Stage 6 normalizes evidence collection across selected runtimes.

## Stage 6: Per-GPU/backend-neutral telemetry

- **Objective:** collect per-device telemetry keyed by physical identity and backend-neutral server-process memory evidence.
- **Affected files:** `hardware.py`, `runner.py`, `context_profile.py`, `progress.py`, `watch.py`, `doctor.py`, `report.py`, `model_cards.py`, templates, tests.
- **Non-goals:** no quality-score weight changes and no inference from telemetry to benchmark correctness.
- **Compatibility constraints:** retain legacy scalar telemetry fields/readers as derived compatibility summaries while adding an explicit `gpus` list. Process memory must identify the selected server process(es) neutrally rather than fields named `ollama_*`; never persist command lines/environments. Distinguish device physical index/UUID/PCI from runtime-visible index.
- **Tests:** multi-device parser fixtures, stable identity joins, selected-process attribution, missing sensor degradation, old telemetry report/card/watch fixtures.
- **Real-hardware acceptance:** collect a multi-GPU selected-profile run and show each device plus selected server memory evidence; verify single-GPU compatibility.
- **Rollback boundary:** consumers can ignore additive list fields and read existing scalar summaries.
- **Documentation outputs:** telemetry schema, attribution limitations, evidence interpretation.
- **Exact next action:** Stage 7 consumes only the new operational evidence in a diagnostic fit lane.

## Stage 7: Runtime-fit profiler

- **Objective:** add a diagnostic lane that records fit/headroom/offload/split observations for a chosen runtime profile without changing quality scores.
- **Affected files:** new fit module/command, `runner.py`, `context_profile.py`, `planner.py`, `report.py`, `model_cards.py`, `watch.py`, tests, docs.
- **Non-goals:** automatic model mutation, Modelfile creation, server reconfiguration, and ranking changes.
- **Compatibility constraints:** use selected profile identity and per-GPU telemetry; preserve current controlled context-profile semantics and safety gates. Fit conclusions must state observed versus configured flags and never claim unmeasured placement.
- **Tests:** deterministic fixture profiles, unavailable telemetry, profile mismatch refusal, no score/ranking mutation, fit report/card rendering.
- **Real-hardware acceptance:** one approved Ollama profile and one external llama-server multi-GPU profile, including explicit-split evidence where present.
- **Rollback boundary:** diagnostic artifacts are separate from canonical rows and can be ignored.
- **Documentation outputs:** fit-lane contract and interpretation guide.
- **Exact next action:** Stage 8 freezes identity through campaigns and reports.

## Stage 8: Campaign/report/ranking/resume integration

- **Objective:** freeze runtime identity in campaign plan/manifest, raw/effective rows, status, reports, cards, candidate rankings, and resume validation.
- **Affected files:** `campaign.py`, `runner.py`, `report.py`, `rankings.py`, `rankings_v3.py`, `rankings_v31.py`, `model_cards.py`, `progress.py`, `watch.py`, `fingerprint.py`, migration/tests/docs.
- **Non-goals:** changes to score formulas, canonical ranking eligibility, or automatic judging/switching beyond profile-aware safety checks.
- **Compatibility constraints:** version schemas additively; legacy run/campaign readers must annotate identity as unavailable rather than invent it. Campaign plan equivalence and resume must reject a changed backend/profile/GPU mapping/server identity before model work. Report duplicate keys and canonical adoption must retain existing score semantics while carrying runtime provenance.
- **Tests:** legacy migration/read compatibility, runtime mismatch resume refusal, plan equivalence identity checks, row/report/card/ranking provenance, package checksum/required-files migration, judge profile isolation, no canonical-score drift.
- **Real-hardware acceptance:** interrupt/resume a bounded selected-runtime campaign and demonstrate refusal after profile identity changes.
- **Rollback boundary:** schema readers accept old evidence; new profile-bearing campaigns stay isolated from old canonical interpretation.
- **Documentation outputs:** schema migration guide, campaign provenance contract, release notes draft.
- **Exact next action:** Stage 9 runs full real acceptance and prepares release review.

## Stage 9: Real acceptance and RC21 release

- **Objective:** validate RC21 on actual configured environments and prepare a reviewable release without overstating evidence.
- **Affected files:** tests, release documentation, changelog/version only after explicit release approval.
- **Non-goals:** RC22 lifecycle/fleet/vision/embedding scope and any automatic deletion.
- **Compatibility constraints:** all baseline tests remain green; scores/canonical rankings remain semantically unchanged; release change is separately approved.
- **Tests:** complete suite, selftest, release check, migration and package verification, clean-tree hygiene checks.
- **Real-hardware acceptance:** healthy Ollama single-GPU selection; healthy llama-server multi-GPU recommendation/selection; ambiguous unattended failure; one-model boundary; separate judge profile/blocker; all-GPU telemetry; resume mismatch refusal; no external-server lifecycle operation.
- **Rollback boundary:** RC21 feature flag/profile opt-out and preserved Ollama compatibility path; no destructive data migration.
- **Documentation outputs:** acceptance matrix, known limitations, operator guide, release notes.
- **Exact next action:** human reviews acceptance evidence and explicitly authorizes version/changelog/release operations.
