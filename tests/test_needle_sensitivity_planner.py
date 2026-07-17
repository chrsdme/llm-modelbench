from llm_modelbench import sensitivity


def test_needle_plan_replaces_non_probe_aligned_ctx_with_probe_aligned_values():
    script = sensitivity.plan_commands(
        run_prefix="needle_probe",
        include_regex="m1|m2",
        tasks="needle",
        level="short",
        ctx_values="default,4096,16384",
        num_predict_values="512",
    )
    assert "auto-promoted: short -> full" in script
    assert "replaced non-probe-aligned needle ctx=4096 with ctx=20000" in script
    assert "replaced non-probe-aligned needle ctx=16384 with ctx=20000" in script
    assert "--level full" in script
    assert "--needle-max-ctx 40960" in script
    assert "--ctx 20000" in script
    assert "--ctx 4096 \\" not in script
    assert "--ctx 8192 \\" not in script
    assert "--ctx 16384 \\" not in script
    assert "needle_probe_default_np512" in script
    assert "needle_probe_ctx20000_np512" in script


def test_needle_plan_drops_unparseable_ctx_and_keeps_probe_aligned_grid():
    script = sensitivity.plan_commands(
        run_prefix="needle_probe",
        include_regex="m1",
        tasks="needle",
        level="full",
        ctx_values="default,bad,2048,8192,40960",
        num_predict_values="512",
    )
    assert "dropped unparseable needle ctx value: bad" in script
    assert "replaced non-probe-aligned needle ctx=2048 with ctx=20000" in script
    assert "replaced non-probe-aligned needle ctx=8192 with ctx=20000" in script
    assert script.count("--ctx 20000") == 1
    assert script.count("--ctx 40960") == 1
    assert "--ctx 2048" not in script
    assert "--ctx 8192 \\" not in script
    assert "--ctx bad" not in script


def test_non_needle_plan_keeps_original_ctx_grid():
    script = sensitivity.plan_commands(
        run_prefix="web_probe",
        include_regex="m1",
        tasks="web_nav",
        level="short",
        ctx_values="default,4096,16384",
        num_predict_values="512",
    )
    assert "--level short" in script
    assert "--ctx 4096" in script
    assert "--ctx 16384" in script
    assert "--needle-max-ctx" not in script
