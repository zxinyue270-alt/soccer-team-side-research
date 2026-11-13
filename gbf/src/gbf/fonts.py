"""Find a CJK-capable system font for Chinese captions on frames and charts.

Fonts are never bundled with gbf and never downloaded at run time: the renderer uses a font
that is already installed on the machine and records which one (family, file name, face
index and the sha256 of the file's raw bytes) in every render manifest, so a rendered
frame can be traced to the exact font bytes that drew it. No absolute path enters a
manifest (:meth:`FontInfo.manifest`).

Search order (the first candidate that passes every check wins):

1. an explicit ``path`` argument;
2. the environment variable ``GBF_CJK_FONT`` (``GBF_CJK_FONT_BOLD`` for the bold weight);
3. the platform's fallback list, :data:`FALLBACK_ORDER`, each file looked up in the
   platform's font directories (:func:`font_directories`):

   - Windows: Microsoft YaHei (``msyh.ttc``), then SimHei, DengXian, SimSun;
   - Linux: Noto Sans CJK SC (``NotoSansCJK-*.ttc`` and the per-language ``.otf``
     packagings), then Noto Sans SC, WenQuanYi Micro Hei / Zen Hei, Droid Sans Fallback;
   - macOS: PingFang SC (``PingFang.ttc``), then Hiragino Sans GB, STHeiti.

An explicit path or environment variable that does not work is an error, never a silent
fallback. A candidate passes when the file exists, Pillow (FreeType) opens the requested
face, and every character of :data:`CJK_PROBE_TEXT` renders as a real glyph rather than
the font's ``.notdef`` box. :class:`FontNotFoundError` lists every candidate tried and why
it was rejected.
"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageDraw, ImageFont

from gbf.util.hashing import sha256_file

Weight = Literal["regular", "bold"]
WEIGHTS: tuple[Weight, ...] = ("regular", "bold")

FONT_ENV_VAR = "GBF_CJK_FONT"
"""Environment variable naming a regular-weight CJK font file (overrides the system list)."""
BOLD_FONT_ENV_VAR = "GBF_CJK_FONT_BOLD"
"""Environment variable naming a bold-weight CJK font file."""

CJK_PROBE_TEXT = "中文高光分析未验证仅画面平面球场速度"
"""Characters a font must render as real glyphs (captions and badges use all of them)."""

_PROBE_SIZE_PX = 24
_NOTDEF_PROBE = chr(0xFFFF)  # a non-character: every font maps it to .notdef
_MAX_FACES = 64

FaceSelector = Callable[[str, str], bool]
"""Predicate on (family name, style name) of one face of a font file or collection."""


class FontNotFoundError(RuntimeError):
    """No usable CJK font was found. ``problems`` lists every candidate and its rejection."""

    def __init__(self, what: str, problems: Sequence[str]) -> None:
        self.what = what
        self.problems = list(problems)
        joined = "\n  - ".join(self.problems) if self.problems else "(no candidates)"
        super().__init__(
            f"{what}: no usable CJK font; install Microsoft YaHei, SimHei, Noto Sans CJK or "
            f"PingFang, or set {FONT_ENV_VAR} to a font file\n  - {joined}"
        )


@dataclass(frozen=True)
class FontCandidate:
    """One entry of a platform's fallback list.

    ``file_name`` is looked up in every font directory of the platform; ``family`` is the
    family name reported in errors and manifests; ``select`` picks the face inside the file
    (a ``.ttc`` collection holds several), ``None`` meaning face 0.
    """

    family: str
    file_name: str
    select: FaceSelector | None = None


@dataclass(frozen=True)
class FontInfo:
    """A font file that passed the checks, with the face to use inside it.

    Attributes:
        path: absolute path of the font file on this machine (never written to manifests).
        sha256: sha256 of the file's raw bytes.
        bytes: file size in bytes.
        family, style: names Pillow reports for the chosen face, e.g. ("Microsoft YaHei",
            "Regular").
        index: face index inside the file (0 for single-face files).
        weight: the weight that was asked for ("regular" or "bold").
        source: where the font came from: "argument", "environment" or "system".
    """

    path: Path
    sha256: str
    bytes: int
    family: str
    style: str
    index: int
    weight: Weight
    source: str

    @property
    def file_name(self) -> str:
        return self.path.name

    def manifest(self) -> dict[str, Any]:
        """JSON form for render manifests: no absolute path, only the name and the hash."""
        return {
            "family": self.family,
            "style": self.style,
            "file_name": self.file_name,
            "index": self.index,
            "weight": self.weight,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "source": self.source,
        }


def _family_is(name: str) -> FaceSelector:
    return lambda family, _style: family == name


def _family_style(name: str, styles: Sequence[str]) -> FaceSelector:
    wanted = {style.lower() for style in styles}
    return lambda family, style: family == name and style.lower() in wanted


def _family_ends(suffix: str) -> FaceSelector:
    return lambda family, _style: family.endswith(suffix)


def _weight_tag(tag: str) -> FaceSelector:
    """Faces whose style is ``tag`` or whose family name ends in it (Hiragino "W3"/"W6")."""
    return lambda family, style: style == tag or family.endswith(tag)


FALLBACK_ORDER: dict[str, dict[Weight, tuple[FontCandidate, ...]]] = {
    "win32": {
        "regular": (
            FontCandidate("Microsoft YaHei", "msyh.ttc", _family_is("Microsoft YaHei")),
            FontCandidate("SimHei", "simhei.ttf"),
            FontCandidate("DengXian", "Deng.ttf"),
            FontCandidate("SimSun", "simsun.ttc", _family_is("SimSun")),
        ),
        "bold": (
            FontCandidate("Microsoft YaHei", "msyhbd.ttc", _family_is("Microsoft YaHei")),
            FontCandidate("SimHei", "simhei.ttf"),
            FontCandidate("DengXian", "Dengb.ttf"),
        ),
    },
    "linux": {
        "regular": (
            FontCandidate("Noto Sans CJK SC", "NotoSansCJK-Regular.ttc", _family_ends("CJK SC")),
            FontCandidate("Noto Sans CJK SC", "NotoSansCJKsc-Regular.otf"),
            FontCandidate("Noto Sans SC", "NotoSansSC-Regular.otf"),
            FontCandidate("Noto Sans SC", "NotoSansSC-Regular.ttf"),
            FontCandidate("WenQuanYi Micro Hei", "wqy-microhei.ttc"),
            FontCandidate("WenQuanYi Zen Hei", "wqy-zenhei.ttc"),
            FontCandidate("Droid Sans Fallback", "DroidSansFallbackFull.ttf"),
        ),
        "bold": (
            FontCandidate("Noto Sans CJK SC", "NotoSansCJK-Bold.ttc", _family_ends("CJK SC")),
            FontCandidate("Noto Sans CJK SC", "NotoSansCJKsc-Bold.otf"),
            FontCandidate("Noto Sans SC", "NotoSansSC-Bold.otf"),
            FontCandidate("Noto Sans SC", "NotoSansSC-Bold.ttf"),
        ),
    },
    "darwin": {
        "regular": (
            FontCandidate(
                "PingFang SC", "PingFang.ttc", _family_style("PingFang SC", ("Regular",))
            ),
            FontCandidate("Hiragino Sans GB", "Hiragino Sans GB.ttc", _weight_tag("W3")),
            FontCandidate("STHeiti", "STHeiti Light.ttc"),
        ),
        "bold": (
            FontCandidate(
                "PingFang SC", "PingFang.ttc", _family_style("PingFang SC", ("Semibold",))
            ),
            FontCandidate("Hiragino Sans GB", "Hiragino Sans GB.ttc", _weight_tag("W6")),
            FontCandidate("STHeiti", "STHeiti Medium.ttc"),
        ),
    },
}
"""Fallback lists per platform key ("win32", "linux", "darwin") and weight, in order."""


def platform_key(platform: str | None = None) -> str:
    """Platform key of the fallback lists: win32, darwin or linux (anything else: linux)."""
    name = sys.platform if platform is None else platform
    if name.startswith("win"):
        return "win32"
    if name == "darwin":
        return "darwin"
    return "linux"


def font_directories(
    platform: str | None = None, environ: Mapping[str, str] | None = None
) -> list[Path]:
    """Directories searched for the fallback list, in order (missing ones are skipped later).

    Windows: ``%WINDIR%/Fonts`` (``%SystemRoot%``, default ``C:/Windows``) and the per-user
    ``%LOCALAPPDATA%/Microsoft/Windows/Fonts``. Linux: ``/usr/share/fonts``,
    ``/usr/local/share/fonts``, ``~/.local/share/fonts`` and ``~/.fonts``, searched
    recursively. macOS: ``/System/Library/Fonts`` (and ``Supplemental``),
    ``/Library/Fonts``, ``~/Library/Fonts`` and, last, the system font asset catalogue
    ``/System/Library/AssetsV2`` where recent macOS versions keep PingFang.
    """
    env = os.environ if environ is None else environ
    key = platform_key(platform)
    if key == "win32":
        windir = env.get("WINDIR") or env.get("SystemRoot") or env.get("SYSTEMROOT") or "C:/Windows"
        dirs = [Path(windir) / "Fonts"]
        local = env.get("LOCALAPPDATA")
        if local:
            dirs.append(Path(local) / "Microsoft" / "Windows" / "Fonts")
        return dirs
    home = Path(env["HOME"]) if env.get("HOME") else Path.home()
    if key == "darwin":
        return [
            Path("/System/Library/Fonts"),
            Path("/System/Library/Fonts/Supplemental"),
            Path("/Library/Fonts"),
            home / "Library" / "Fonts",
            Path("/System/Library/AssetsV2"),
        ]
    return [
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        home / ".local" / "share" / "fonts",
        home / ".fonts",
    ]


def _locate(file_name: str, directories: Sequence[Path], recursive: bool) -> list[Path]:
    found: list[Path] = []
    for directory in directories:
        if not directory.is_dir():
            continue
        direct = directory / file_name
        if direct.is_file():
            found.append(direct)
            continue
        if recursive:
            for root, _dirs, files in os.walk(directory):
                if file_name in files:
                    found.append(Path(root) / file_name)
                    break
    return found


def _open_face(path: str, size_px: int, index: int) -> ImageFont.FreeTypeFont:
    """Open one face of the font file at ``path`` and nothing else.

    ``ImageFont.truetype`` must not be used here: when a file fails to load it silently
    retries with a file of the same name from the system font directory, so a broken
    ``msyh.ttc`` elsewhere would "load" (and the manifest would hash the wrong bytes).
    ``FreeTypeFont`` raises OSError instead.
    """
    return ImageFont.FreeTypeFont(path, size_px, index=index)


def _faces(path: Path) -> list[tuple[int, str, str]]:
    """(index, family, style) of every face Pillow can open in ``path``."""
    faces: list[tuple[int, str, str]] = []
    for index in range(_MAX_FACES):
        try:
            font = _open_face(str(path), _PROBE_SIZE_PX, index)
        except OSError:
            break
        family, style = font.getname()
        faces.append((index, str(family or ""), str(style or "")))
    return faces


def _glyph_signature(font: ImageFont.FreeTypeFont, char: str) -> bytes | None:
    """Rendered bitmap of one character (None when it draws nothing)."""
    left, top, right, bottom = font.getbbox(char)
    width, height = int(right - left), int(bottom - top)
    if width <= 0 or height <= 0:
        return None
    image = Image.new("L", (width, height), 0)
    ImageDraw.Draw(image).text((-left, -top), char, font=font, fill=255)
    if image.getbbox() is None:
        return None
    return f"{width}x{height}:".encode("ascii") + image.tobytes()


def missing_glyphs(font: ImageFont.FreeTypeFont, text: str = CJK_PROBE_TEXT) -> list[str]:
    """Characters of ``text`` that ``font`` draws as nothing or as its ``.notdef`` glyph."""
    notdef = _glyph_signature(font, _NOTDEF_PROBE)
    missing: list[str] = []
    for char in dict.fromkeys(text):
        if char.isspace():
            continue
        signature = _glyph_signature(font, char)
        if signature is None or signature == notdef:
            missing.append(char)
    return missing


@lru_cache(maxsize=32)
def _hash_cached(path: str, size: int, mtime_ns: int) -> str:
    return sha256_file(path)


def _file_hash(path: Path) -> tuple[str, int]:
    stat = path.stat()
    return _hash_cached(str(path), stat.st_size, stat.st_mtime_ns), stat.st_size


def _check_candidate(
    path: Path,
    select: FaceSelector | None,
    family_hint: str,
    weight: Weight,
    source: str,
    check_glyphs: bool,
) -> tuple[FontInfo | None, str | None]:
    """(FontInfo, None) when ``path`` passes, else (None, reason)."""
    if not path.is_file():
        return None, f"{family_hint}: {path} does not exist"
    faces = _faces(path)
    if not faces:
        return None, f"{family_hint}: {path} cannot be opened by Pillow/FreeType"
    chosen = faces[0]
    if select is not None:
        matching = [face for face in faces if select(face[1], face[2])]
        if not matching:
            names = ", ".join(f"{family} {style}".strip() for _, family, style in faces)
            return None, f"{family_hint}: no matching face in {path.name} (faces: {names})"
        chosen = matching[0]
    index, family, style = chosen
    if check_glyphs:
        font = _open_face(str(path), _PROBE_SIZE_PX, index)
        missing = missing_glyphs(font)
        if missing:
            return None, (
                f"{family_hint}: {path.name} face {index} ({family} {style}) has no glyph for "
                f"{''.join(missing)}"
            )
    sha256, size = _file_hash(path)
    info = FontInfo(
        path=path.resolve(),
        sha256=sha256,
        bytes=size,
        family=family or family_hint,
        style=style,
        index=index,
        weight=weight,
        source=source,
    )
    return info, None


def find_cjk_font(
    weight: Weight = "regular",
    *,
    path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    platform: str | None = None,
    search_dirs: Sequence[str | Path] | None = None,
    check_glyphs: bool = True,
) -> FontInfo:
    """Return the first usable CJK font for ``weight`` (see the module docstring for the order).

    Args:
        weight: "regular" or "bold".
        path: an explicit font file; when given, it must work (no fallback).
        environ: environment to read ``GBF_CJK_FONT`` / ``GBF_CJK_FONT_BOLD`` and the font
            directories from (default ``os.environ``).
        platform: platform name as in ``sys.platform`` (default: this machine's).
        search_dirs: directories to search instead of :func:`font_directories`, searched
            recursively.
        check_glyphs: verify that every character of :data:`CJK_PROBE_TEXT` has a glyph.

    Raises:
        ValueError: ``weight`` is not "regular" or "bold".
        FontNotFoundError: no candidate passed; lists every candidate and the reason.
    """
    if weight not in WEIGHTS:
        raise ValueError(f"weight must be one of {WEIGHTS}, got {weight!r}")
    env = os.environ if environ is None else environ
    what = f"find_cjk_font({weight})"
    if path is not None:
        info, reason = _check_candidate(
            Path(path), None, "explicit path", weight, "argument", check_glyphs
        )
        if info is None:
            raise FontNotFoundError(what, [str(reason)])
        return info
    variable = FONT_ENV_VAR if weight == "regular" else BOLD_FONT_ENV_VAR
    configured = env.get(variable)
    if configured:
        info, reason = _check_candidate(
            Path(configured), None, variable, weight, "environment", check_glyphs
        )
        if info is None:
            raise FontNotFoundError(what, [f"{variable} is set but unusable: {reason}"])
        return info

    key = platform_key(platform)
    if search_dirs is not None:
        directories = [Path(directory) for directory in search_dirs]
        recursive = True
    else:
        directories = font_directories(key, env)
        recursive = key != "win32"
    problems: list[str] = []
    for candidate in FALLBACK_ORDER[key][weight]:
        located = _locate(candidate.file_name, directories, recursive)
        if not located:
            searched = ", ".join(str(directory) for directory in directories)
            problems.append(f"{candidate.family}: {candidate.file_name} not found in {searched}")
            continue
        for found in located:
            info, reason = _check_candidate(
                found, candidate.select, candidate.family, weight, "system", check_glyphs
            )
            if info is not None:
                return info
            problems.append(str(reason))
    raise FontNotFoundError(what, problems)


def find_cjk_fonts(
    *,
    environ: Mapping[str, str] | None = None,
    platform: str | None = None,
    search_dirs: Sequence[str | Path] | None = None,
) -> tuple[FontInfo, FontInfo | None]:
    """(regular, bold or None): the bold weight is optional, the regular one is required."""
    regular = find_cjk_font("regular", environ=environ, platform=platform, search_dirs=search_dirs)
    try:
        bold = find_cjk_font("bold", environ=environ, platform=platform, search_dirs=search_dirs)
    except FontNotFoundError:
        bold = None
    return regular, bold


@lru_cache(maxsize=256)
def _truetype(path: str, index: int, size_px: int) -> ImageFont.FreeTypeFont:
    return _open_face(path, size_px, index)


def load_font(info: FontInfo, size_px: float) -> ImageFont.FreeTypeFont:
    """Pillow font object for ``info`` at ``size_px`` pixels (em size), cached.

    Raises ValueError when ``size_px`` is not a positive finite number.
    """
    if (
        isinstance(size_px, bool)
        or not isinstance(size_px, int | float)
        or not math.isfinite(size_px)
        or size_px <= 0
    ):
        raise ValueError(f"size_px must be a positive finite number, got {size_px!r}")
    return _truetype(str(info.path), info.index, max(1, int(round(size_px))))
