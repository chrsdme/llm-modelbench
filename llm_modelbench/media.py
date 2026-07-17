"""Image rendering for OCR/PDF tasks. Optional: needs Pillow.

Renders known text to a PNG so ground truth is exact and reproducible. A noisy variant adds
blur and speckle to approximate a real scan. If Pillow is not installed, these return None and
the runner skips vision tasks with a clear reason rather than crashing. Drop your own labelled
scans in and point a task's reference at their exact text for real-world OCR.
"""
from __future__ import annotations

import base64
import mimetypes
import random
import tempfile
import textwrap
from pathlib import Path
from typing import Dict, Optional, Union


_IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def load_image_file(path: Union[str, Path]) -> Dict[str, str]:
    """Load a task image fixture as base64 for the existing Ollama image payload.

    Relative paths are resolved from the repository root so task metadata is stable
    regardless of the command's working directory.
    """
    image_path = Path(path).expanduser()
    if not image_path.is_absolute():
        package_candidate = Path(__file__).resolve().parent / image_path
        repository_candidate = Path(__file__).resolve().parent.parent / image_path
        image_path = package_candidate if package_candidate.exists() else repository_candidate
    image_path = image_path.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"image fixture not found: {image_path}")

    suffix = image_path.suffix.lower()
    mime_type = _IMAGE_MIME_TYPES.get(suffix)
    if mime_type is None:
        guessed, _ = mimetypes.guess_type(str(image_path))
        raise ValueError(f"unsupported image fixture type {suffix or '(none)'}: {guessed or 'unknown'}")

    data = image_path.read_bytes()
    if not data:
        raise ValueError(f"image fixture is empty: {image_path}")
    return {
        "data": base64.b64encode(data).decode(),
        "mime_type": mime_type,
        "path": str(image_path),
    }


def render_text_png(text: str, noisy: bool = False, seed: int = 42) -> Optional[str]:
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageFilter
    except Exception:
        return None
    lines = []
    for para in text.split("\n"):
        lines.extend(textwrap.wrap(para, 46) or [""])
    width, line_h, pad = 900, 42, 28
    img = Image.new("RGB", (width, pad * 2 + line_h * max(1, len(lines))), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
    except Exception:
        font = ImageFont.load_default()
    y = pad
    for ln in lines:
        d.text((pad, y), ln, fill="black", font=font)
        y += line_h
    if noisy:
        img = img.filter(ImageFilter.GaussianBlur(0.6))
        px = img.load()
        random.seed(seed)
        for _ in range(int(img.size[0] * img.size[1] * 0.02)):
            x, yy = random.randint(0, img.size[0] - 1), random.randint(0, img.size[1] - 1)
            px[x, yy] = (random.randint(0, 90),) * 3
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = Path(f.name)
    img.save(tmp)
    b64 = base64.b64encode(tmp.read_bytes()).decode()
    tmp.unlink(missing_ok=True)
    return b64
