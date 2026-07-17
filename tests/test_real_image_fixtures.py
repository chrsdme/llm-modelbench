import base64

import pytest

from llm_modelbench import media, runner
from llm_modelbench.config import Config
from llm_modelbench.tasks import Task


_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLJ6QAAAABJRU5ErkJggg=="
)


def _vision_task(**meta):
    base_meta = {"reference": "fixture text"}
    base_meta.update(meta)
    return Task("fixture_ocr", "ocr", "vision", "ocr", "unused", meta=base_meta)


def test_load_image_file_returns_base64_media_metadata(tmp_path):
    image_path = tmp_path / "ocr_tiny.png"
    image_path.write_bytes(_TINY_PNG)

    payload = media.load_image_file(image_path)

    assert payload["mime_type"] == "image/png"
    assert payload["path"] == str(image_path.resolve())
    assert base64.b64decode(payload["data"]) == _TINY_PNG


def test_load_image_file_rejects_missing_and_unsupported_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="image fixture not found"):
        media.load_image_file(tmp_path / "missing.png")

    unsupported = tmp_path / "image.gif"
    unsupported.write_bytes(b"GIF89a")
    with pytest.raises(ValueError, match="unsupported image fixture type"):
        media.load_image_file(unsupported)


def test_runner_uses_real_image_fixture_payload(monkeypatch, tmp_path):
    image_path = tmp_path / "ocr_tiny.png"
    image_path.write_bytes(_TINY_PNG)
    task = _vision_task(image_path=str(image_path))
    captured = {}

    monkeypatch.setattr(runner.media, "render_text_png", lambda *_: pytest.fail("synthetic image used"))

    def fake_chat(*_args, **kwargs):
        captured["images"] = kwargs["images"]
        return {"ok": True, "text": "fixture text"}

    monkeypatch.setattr(runner, "_chat", fake_chat)

    runner._run_once(object(), Config(), "unused", task)

    assert captured["images"] == [base64.b64encode(_TINY_PNG).decode()]


def test_runner_without_image_path_keeps_synthetic_payload(monkeypatch):
    task = _vision_task()
    captured = {}

    monkeypatch.setattr(runner.media, "render_text_png", lambda *_: "synthetic-base64")

    def fake_chat(*_args, **kwargs):
        captured["images"] = kwargs["images"]
        return {"ok": True, "text": "fixture text"}

    monkeypatch.setattr(runner, "_chat", fake_chat)

    runner._run_once(object(), Config(), "unused", task)

    assert captured["images"] == ["synthetic-base64"]


def test_runner_reports_missing_image_before_model_call(monkeypatch, tmp_path):
    task = _vision_task(image_path=str(tmp_path / "missing.png"))
    monkeypatch.setattr(runner, "_chat", lambda *_args, **_kwargs: pytest.fail("model called"))

    result = runner._run_once(object(), Config(), "unused", task)

    assert result["error_kind"] == "harness_error"
    assert "vision image fixture error" in result["reason"]
