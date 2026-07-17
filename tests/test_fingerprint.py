from llm_modelbench.fingerprint import similarity, find_clones, find_digest_clones, invalid_probe_models


def test_empty_probe_outputs_do_not_clone():
    a = ["", "", "", "", "", ""]
    b = ["", "", "", "", "", ""]
    assert similarity(a, b) == 0.0
    assert find_clones({"a": a, "b": b}) == []


def test_near_empty_outputs_do_not_clone():
    a = ["ok", "yes", "no", "-", "1", "2"]
    b = ["ok", "yes", "no", "-", "1", "2"]
    assert similarity(a, b) == 0.0


def test_exact_valid_probe_outputs_can_clone():
    outs = [
        "This benchmark can overfit when tasks are too narrow. Use held-out tasks.",
        '{"risk":"bad","cause":"namespace","fix":"serialize primitives"}',
        "def stable_slug(text):\n    return text.lower().strip('-')",
        "Long context can be slower because KV cache grows with sequence length.",
        "ollama,benchmark,local-ai",
        "find /srv/ai-workdesk -xdev -type f -printf '%s %p\\n' | sort -nr | head",
    ]
    assert similarity(outs, list(outs)) == 1.0
    assert find_clones({"a": outs, "b": list(outs)}) == [("a", "b", 1.0)]


def test_insufficient_valid_probe_pairs_do_not_clone():
    a = ["", "", "", "", "", "valid output long enough to count"]
    b = ["", "", "", "", "", "valid output long enough to count"]
    assert similarity(a, b) == 0.0
    bad = invalid_probe_models({"a": a})
    assert bad["a"]["valid"] == 1


def test_digest_clones_are_exact_identity_only():
    ids = {
        "a": {"digest": "abc123456789"},
        "b": {"digest": "abc123456789"},
        "c": {"digest": "zzz987654321"},
        "d": {"digest": ""},
    }
    assert find_digest_clones(ids) == [("a", "b", 1.0)]
