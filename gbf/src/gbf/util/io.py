"""File IO that behaves the same on Windows and Linux.

- Text is always UTF-8 with LF line endings (Python's text mode would write CRLF on Windows).
- Images go through Pillow, which handles non-ASCII paths (OpenCV image IO does not).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def write_json(path: str | Path, obj: Any, *, indent: int = 2) -> None:
    text = json.dumps(obj, indent=indent, ensure_ascii=False, sort_keys=True, allow_nan=False)
    write_text(path, text + "\n")


def read_json(path: str | Path) -> Any:
    with open(path, "rb") as fh:
        return json.loads(fh.read().decode("utf-8"))


def read_image_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def write_image_png(path: str | Path, array: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(array)).save(path, format="PNG")
