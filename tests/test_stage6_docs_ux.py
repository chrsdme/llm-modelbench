from pathlib import Path

import pytest

from llm_modelbench import cli


ROOT = Path(__file__).resolve().parents[1]


def _doc(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_stage6_acceptance_controls_document_verified_workflows():
    text = _doc("docs/ACCEPTANCE_CONTROLS.md")
    required = [
        "campaign init campaign.json",
        "campaign execute --config campaign.json --mock",
        "campaign supersede",
        "unsupported schema versions",
        "forks such as `A -> B` and `A -> C`",
        "cycles all",
        "immutable config plan signature",
        "ready_for_adoption",
        "do not rewrite source raw rows",
    ]
    for phrase in required:
        assert phrase in text
    assert "real Selene qualification happened" not in text
    assert "production acceptance" not in text


def test_stage6_public_docs_link_acceptance_controls():
    assert "docs/ACCEPTANCE_CONTROLS.md" in _doc("README.md")
    assert "ACCEPTANCE_CONTROLS.md" in _doc("docs/USAGE.md")
    assert "ACCEPTANCE_CONTROLS.md" in _doc("docs/CAMPAIGNS.md")
    assert "ACCEPTANCE_CONTROLS.md" in _doc("docs/README.md")


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["campaign", "--help"], "supersession appends validated evidence"),
        (["campaign", "execute", "--help"], "immutable plan signature"),
        (["campaign", "supersede", "--help"], "forks, cycles"),
    ],
)
def test_stage6_campaign_help_names_acceptance_controls(argv, expected, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 0
    assert expected in capsys.readouterr().out
