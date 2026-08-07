"""Service ownership is discovered and checked by the root broker, not Python."""
import pytest

from llm_modelbench.ollama_service import ServiceControlError, discover_active_service


def test_unprivileged_discovery_is_explicitly_not_a_control_path():
    with pytest.raises(ServiceControlError, match="privileged KV broker"):
        discover_active_service()
