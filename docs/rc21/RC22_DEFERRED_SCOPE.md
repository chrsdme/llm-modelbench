# RC22 Deferred Scope

RC21 supports externally managed `llama-server` only. RC22 requirements are deliberately deferred and require separate design, approval, and safety review:

- managed automatic `llama-server` launch, readiness handling, cleanup, and lifecycle ownership;
- automatic Ollama Modelfile generation from validated runtime profiles;
- permanent user-level systemd model services;
- permanent runtime watcher/session logger;
- non-destructive fleet classification and an explicit-approval removal workflow;
- vision/OCR benchmark expansion; and
- embedding benchmark changes.

No automatic model deletion is permitted in RC21 or RC22. A future removal workflow must produce a non-destructive inventory/classification preview, require a typed operator approval, retain an auditable decision record, and leave runtime management separate from benchmark scoring.

Automated GGUF fleet switching is also RC22 work. RC21 can use one served model from one external endpoint/profile; multiple preconfigured endpoints are distinct profiles, not a model-switching mechanism.
