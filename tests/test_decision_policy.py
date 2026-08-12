import pytest

from llm_modelbench.decision_policy import Action, DecisionPolicy, DEFAULT_POLICY


def test_default_policy_permits_nothing():
    assert DEFAULT_POLICY.unattended is False
    assert DEFAULT_POLICY.permits(Action.BACKEND_AUTO_SELECT) is False


def test_unattended_alone_does_not_imply_backend_auto_select():
    policy = DecisionPolicy(unattended=True)
    assert policy.permits(Action.BACKEND_AUTO_SELECT) is False


def test_backend_auto_select_requires_both_unattended_and_explicit_grant():
    assert DecisionPolicy(unattended=False, allow_backend_auto_selection=True).permits(Action.BACKEND_AUTO_SELECT) is False
    assert DecisionPolicy(unattended=True, allow_backend_auto_selection=True).permits(Action.BACKEND_AUTO_SELECT) is True


def test_policy_is_frozen():
    policy = DecisionPolicy()
    with pytest.raises(Exception):
        policy.unattended = True
