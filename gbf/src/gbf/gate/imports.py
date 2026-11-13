"""License-boundary checks for P0 (gate G3): forbidden imports and denied distributions.

The banned names follow registries/models.toml: every entry whose status is blocked,
research_only or plugin_only (BANNED_STATUSES) has a row in REGISTRY_BANS, keyed by its
registry id, with the top-level modules it is imported as and the distributions it is
installed as. The registry does not record those names, so the rows are written by hand;
an entry that is not a package lists none and says why in its note. FORBIDDEN_MODULES
(plus the pickle family: no pickle for artifacts, ADR 0003) and DENIED_DISTRIBUTIONS are
the unions of the rows.

forbidden_imports parses every ``.py``, ``.pyi`` or ``.pyw`` candidate file under
``src/`` and ``tests/`` and fails on every way the file names a forbidden top-level
module:

- absolute ``import x`` / ``from x import y`` anywhere in the file (also inside a
  function);
- a call of a loader (LOADERS) with a literal module name, positional or as a keyword
  (``name=``, ``modname=``, ``fullname=``, ``mod_name=``, ``module_or_name=``):
  ``importlib.import_module``, ``__import__`` (also ``builtins.__import__`` and
  ``importlib.__import__``), ``pytest.importorskip``, ``importlib.util.find_spec`` (and a
  finder's ``find_spec``), ``pkgutil.get_loader`` and ``pkgutil.find_loader`` (which
  locate the module for a loader), ``runpy.run_module`` and ``pkgutil.resolve_name``
  (``"module:attribute"``). A loader is recognised by its attribute name whatever object
  it is reached through (``il.import_module`` after ``import importlib as il``), by its
  bare name, and by an alias bound anywhere in the file (``from importlib import
  import_module as im``, ``load = importlib.import_module``, also chained, unpacked from a
  tuple as in ``load, n = importlib.import_module, 1``, or bound and called at once as in
  ``(load := importlib.import_module)(...)``). A literal name is a string constant,
  literal pieces joined with ``+`` or an f-string without fields; a relative name
  (``".engine"``) given to import_module or find_spec counts as its literal package
  argument.

Names computed at run time are out of scope: variables, string formatting or joining
(also an f-string field), conditional expressions, names passed through ``*args`` or
``**kwargs``, loaders reached through ``getattr``, a namespace lookup such as
``builtins.__dict__["__import__"]``, ``functools.partial`` or a wrapper function, loaders
kept in an attribute (``self.load = importlib.import_module``) or a default argument,
and code in strings passed to ``exec``/``eval``. A test that checks that a forbidden
module is absent (``find_spec(...) is None``) therefore builds its name at run time.
Plugins run out of process (ADR 0001), so plugin code lives outside these trees and may
import what it needs.

A file that does not parse fails as ``syntax-error``. The check also reads
registries/models.toml, when the repository has one, and fails as
``unmapped-registry-entry`` (detail: the id) for every entry with a banned status that
REGISTRY_BANS does not list, so a newly blocked component cannot slip past the lists.

deps fails on denied distributions (names normalised as in PEP 503) in pyproject.toml
(project dependencies, optional dependencies, dependency groups, tool.uv dev
dependencies and sources, build requirements, and direct-reference URLs) and in the
packages of uv.lock. It is a deny list only: full license-metadata scanning
(pip-licenses, REUSE, NOTICES) arrives at the P5 gate.
"""

from __future__ import annotations

import ast
import re
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from gbf.gate.model import CheckResult
from gbf.gate.repo import Workspace, toml_error_line


def normalize_name(name: str) -> str:
    """PEP 503 normalised distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass(frozen=True)
class RegistryBan:
    """The names under which one registries/models.toml entry could enter the core.

    ``imports`` are top-level module names, ``distributions`` package names as published
    (compared after PEP 503 normalisation). ``note`` says why an entry lists no names, or
    where a name comes from when that is not obvious."""

    imports: tuple[str, ...] = ()
    distributions: tuple[str, ...] = ()
    note: str = ""


MODELS_REGISTRY = "registries/models.toml"
#: Registry statuses whose entries the core never imports or depends on (ADR 0001).
BANNED_STATUSES = ("blocked", "research_only", "plugin_only")
#: One row per registries/models.toml entry with a BANNED_STATUSES status, by registry id.
REGISTRY_BANS: dict[str, RegistryBan] = {
    # plugin_only
    "sam-3d-body": RegistryBan(("sam_3d_body",), ("sam-3d-body",)),
    "plugin-sam3d-patches": RegistryBan(("sam_3d_body",), note="patches sam_3d_body sources"),
    "sn-gamestate": RegistryBan(("sn_gamestate",), ("sn-gamestate",)),
    "tracklab": RegistryBan(("tracklab",), ("tracklab",)),
    "ultralytics-yolo11": RegistryBan(("ultralytics",), ("ultralytics",)),
    "prtreid": RegistryBan(("prtreid",), ("prtreid",)),
    "bpbreid": RegistryBan(
        ("torchreid", "bpbreid"), ("torchreid", "bpbreid"), note="installed as torchreid"
    ),
    "mmocr": RegistryBan(
        ("mmocr", "mmdet", "mmcv", "mmengine"),
        ("mmocr", "mmdet", "mmcv", "mmcv-full", "mmcv-lite", "mmengine"),
        note="with the OpenMMLab packages it requires",
    ),
    "nbjw-calib": RegistryBan(
        note="not a package: vendored inside sn-gamestate (banned as sn_gamestate); "
        "plugin-nbjw runs the original repository out of process"
    ),
    "pnlcalib": RegistryBan(note="not a package: bundled inside sn-gamestate"),
    "easyocr": RegistryBan(
        ("easyocr", "bidi"), ("easyocr", "python-bidi"), note="python-bidi imports as bidi"
    ),
    # research_only
    "nlf-l": RegistryBan(("nlf",), ("nlf",)),
    "rtmw3d": RegistryBan(
        ("mmpose", "rtmpose3d", "mmcv", "mmengine"),
        ("mmpose", "mmcv", "mmengine"),
        note="mmpose's rtmpose3d project and the OpenMMLab packages it requires",
    ),
    "field-converter": RegistryBan(
        ("field_converter",),
        ("field-converter",),
        note="names follow the repository; no published package was checked",
    ),
    "smpl-family": RegistryBan(("smplx",), ("smplx",)),
    # blocked
    "fast-sam-3d-body": RegistryBan(
        ("sam_3d_body",),
        note="not published as a package; it ships a sam_3d_body tree and needs "
        "ultralytics (banned as ultralytics-yolo11)",
    ),
    "rf-detr-plus": RegistryBan(("rfdetr_plus",), ("rfdetr-plus", "rf-detr-plus")),
    "tvcalib-segmentation": RegistryBan(
        ("sn_segmentation",), note="a git submodule of TVCalib, not published as a package"
    ),
    "sn-calibration": RegistryBan(note="code only, not a package"),
    "sn-tracking-code": RegistryBan(note="code only, not a package"),
    "sn-jersey": RegistryBan(note="the repository holds only a README"),
    "deep-eiou": RegistryBan(note="code only, not a package"),
    "learning-from-the-pros": RegistryBan(note="code only, not a package"),
    "jersey-number-pipeline": RegistryBan(note="code only, not a package"),
    "deimv2": RegistryBan(
        ("deimv2",),
        ("deimv2",),
        note="names follow the repository; no published package was checked",
    ),
    "boxmot": RegistryBan(("boxmot",), ("boxmot",)),
    "roboflow-sports-example-weights": RegistryBan(
        note="weight files: ultralytics is banned as ultralytics-yolo11, and the MIT "
        "sports package itself is allowed"
    ),
}
#: The pickle family (no pickle for artifacts, ADR 0003).
PICKLE_MODULES = frozenset({"pickle", "_pickle", "cPickle", "cloudpickle", "dill"})
FORBIDDEN_MODULES = (
    frozenset(name for ban in REGISTRY_BANS.values() for name in ban.imports) | PICKLE_MODULES
)
DENIED_DISTRIBUTIONS = frozenset(
    normalize_name(name) for ban in REGISTRY_BANS.values() for name in ban.distributions
)
P5_NOTE = (
    "deny list only; full license-metadata scanning (pip-licenses, REUSE, NOTICES) "
    "arrives at the P5 gate"
)
SCANNED_TREES = ("src/", "tests/")
#: Python source files, also the Windows .pyw scripts.
SCANNED_SUFFIXES = frozenset({".py", ".pyi", ".pyw"})
#: Functions that import, or locate for a loader, the module named by their first
#: argument (see the module docstring).
LOADERS = frozenset(
    {
        "import_module",
        "__import__",
        "importorskip",
        "find_spec",
        "get_loader",
        "find_loader",
        "run_module",
        "resolve_name",
    }
)
#: Keyword names of that argument in the loaders' signatures.
_NAME_KEYWORDS = ("name", "modname", "fullname", "mod_name", "module_or_name")
#: Loaders that resolve a relative name (".x") against their package argument.
_PACKAGE_LOADERS = frozenset({"import_module", "find_spec"})

_REQUIREMENT_NAME = re.compile(r"\s*([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)")
_URL_SPLIT = re.compile(r"[/#@=?&:+\s\"',\[\]{}]+")
_ARCHIVE_NAME = re.compile(r"([A-Za-z0-9][A-Za-z0-9._]*?)-\d")
_QUOTED_NAME = re.compile(r"[\"']\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
_REGISTRY_ID = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")


def unmapped_registry_ids(registry: Any) -> list[str]:
    """Sorted ids of model-registry entries with a BANNED_STATUSES status and no ban row."""
    models = registry.get("models") if isinstance(registry, dict) else None
    found: set[str] = set()
    for entry in models if isinstance(models, list) else []:
        if not isinstance(entry, dict) or entry.get("status") not in BANNED_STATUSES:
            continue
        entry_id = entry.get("id")
        if isinstance(entry_id, str) and entry_id not in REGISTRY_BANS:
            found.add(entry_id)
    return sorted(found)


def _literal(node: ast.expr | None) -> str | None:
    """The value of a literal string: a constant, literal pieces joined with ``+``, or an
    f-string without fields. Anything computed at run time gives None."""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.JoinedStr):
        pieces = [_literal(value) for value in node.values]
        return None if None in pieces else "".join(pieces)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _literal(node.left), _literal(node.right)
        return None if left is None or right is None else left + right
    return None


def _loader(func: ast.expr, names: dict[str, str]) -> str | None:
    """The loader an expression refers to: an attribute named like one (whatever object it
    is reached through), a name bound to one in ``names``, or an assignment expression
    whose value is one of these."""
    if isinstance(func, ast.NamedExpr):
        return _loader(func.value, names)
    if isinstance(func, ast.Attribute):
        return func.attr if func.attr in LOADERS else None
    if isinstance(func, ast.Name):
        return names.get(func.id)
    return None


def _bindings(target: ast.expr, value: ast.expr) -> list[tuple[str, ast.expr]]:
    """(name, value) pairs that assigning ``value`` to ``target`` binds where the value is
    a name or an attribute, also element by element when a tuple or list is unpacked from
    one of the same length (``load, n = importlib.import_module, 1``)."""
    if isinstance(target, ast.Name) and isinstance(value, ast.Name | ast.Attribute):
        return [(target.id, value)]
    if (
        isinstance(target, ast.Tuple | ast.List)
        and isinstance(value, ast.Tuple | ast.List)
        and len(target.elts) == len(value.elts)
        and not any(isinstance(e, ast.Starred) for e in [*target.elts, *value.elts])
    ):
        pairs = zip(target.elts, value.elts, strict=True)
        return [pair for t, v in pairs for pair in _bindings(t, v)]
    return []


def _loader_names(tree: ast.AST) -> dict[str, str]:
    """Names that refer to a loader in this file, with the loader each one refers to: the
    loaders' own names and every alias bound by ``from ... import <loader> as <alias>`` or
    by assigning a loader (or another alias) to a name, wherever that happens."""
    names = {loader: loader for loader in LOADERS}
    bindings: list[tuple[str, ast.expr]] = []  # name = <a name or an attribute>
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in LOADERS:
                    names[alias.asname or alias.name] = alias.name
            continue
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign | ast.NamedExpr) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        bindings += [pair for target in targets for pair in _bindings(target, value)]
    grown = True
    while grown:  # aliases of aliases; each pass adds a name or ends the loop
        grown = False
        for name, value in bindings:
            loader = _loader(value, names)
            if loader is not None and name not in names:
                names[name] = loader
                grown = True
    return names


def _loaded_module(call: ast.Call, loader: str) -> str | None:
    """The module a loader call names literally, or None when the name is computed."""
    keywords = {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg}
    if call.args:
        name = _literal(call.args[0])
    else:
        name = next((_literal(keywords[k]) for k in _NAME_KEYWORDS if k in keywords), None)
    if name is not None and name.startswith(".") and loader in _PACKAGE_LOADERS:
        package = call.args[1] if len(call.args) > 1 else keywords.get("package")
        return _literal(package)
    return name


def _imports(tree: ast.AST) -> Iterator[tuple[str, int]]:
    """(top-level module, line) of every import and literal loader call in ``tree``."""
    loaders = _loader_names(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                yield node.module.split(".")[0], node.lineno
        elif isinstance(node, ast.Call):
            loader = _loader(node.func, loaders)
            module = _loaded_module(node, loader) if loader is not None else None
            if module:
                yield re.split(r"[.:]", module)[0], node.lineno


def _id_line(text: str, entry_id: str) -> int | None:
    """Line of ``id = "<entry_id>"`` in a registry's TOML text."""
    pattern = re.compile(rf"\s*id\s*=\s*[\"']{re.escape(entry_id)}[\"']")
    for number, line in enumerate(text.splitlines(), start=1):
        if pattern.match(line):
            return number
    return None


def _check_registry_rows(ws: Workspace, result: CheckResult) -> None:
    """Fail every banned entry of registries/models.toml that REGISTRY_BANS does not list."""
    if MODELS_REGISTRY not in ws.file_set:
        return
    loaded = _load_toml(ws, MODELS_REGISTRY, result)
    if loaded is None:
        return
    data, text = loaded
    result.scanned += 1
    for entry_id in unmapped_registry_ids(data):
        safe = entry_id if _REGISTRY_ID.fullmatch(entry_id) else "?"
        line = _id_line(text, entry_id)
        result.add(MODELS_REGISTRY, "unmapped-registry-entry", line=line, detail=safe)


def check_forbidden_imports(root: Path, workspace: Workspace | None = None) -> CheckResult:
    """Fail imports of banned or pickle-family modules in src/ and tests/, and banned
    entries of registries/models.toml that have no row in REGISTRY_BANS."""
    ws = workspace if workspace is not None else Workspace(root)
    result = CheckResult("forbidden_imports")
    _check_registry_rows(ws, result)
    for rel in ws.files:
        if not rel.startswith(SCANNED_TREES) or PurePosixPath(rel).suffix not in SCANNED_SUFFIXES:
            continue
        try:
            source = ws.read(rel)
        except OSError as error:
            result.add(rel, "unreadable", detail=type(error).__name__)
            continue
        if source is None:
            continue
        result.scanned += 1
        try:
            tree = ast.parse(source, filename=rel)
        except SyntaxError as error:
            result.add(rel, "syntax-error", line=error.lineno)
            continue
        except ValueError:
            result.add(rel, "syntax-error")
            continue
        for module, line in _imports(tree):
            if module in FORBIDDEN_MODULES:
                result.add(rel, module, line=line)
    return result


def requirement_names(requirement: str) -> set[str]:
    """Normalised names a PEP 508 requirement refers to, including names in a direct URL."""
    names: set[str] = set()
    match = _REQUIREMENT_NAME.match(requirement)
    if match:
        names.add(normalize_name(match.group(1)))
    if "@" in requirement:
        url = requirement.split("@", 1)[1]
        for segment in _URL_SPLIT.split(url):
            if not segment:
                continue
            for ending in (".git", ".whl", ".zip", ".tar.gz"):
                segment = segment.removesuffix(ending)
            names.add(normalize_name(segment))
            archive = _ARCHIVE_NAME.match(segment)
            if archive:
                names.add(normalize_name(archive.group(1)))
    return names


def _string_list(value: Any) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def pyproject_requirements(data: dict[str, Any]) -> list[tuple[str, str]]:
    """(where, requirement) pairs from every dependency declaration in pyproject data."""
    found: list[tuple[str, str]] = []
    project = data.get("project") if isinstance(data.get("project"), dict) else {}
    found += [("project.dependencies", r) for r in _string_list(project.get("dependencies"))]
    optional = project.get("optional-dependencies")
    if isinstance(optional, dict):
        for extra, items in optional.items():
            found += [(f"project.optional-dependencies.{extra}", r) for r in _string_list(items)]
    groups = data.get("dependency-groups")
    if isinstance(groups, dict):
        for group, items in groups.items():
            found += [(f"dependency-groups.{group}", r) for r in _string_list(items)]
    build = data.get("build-system")
    if isinstance(build, dict):
        found += [("build-system.requires", r) for r in _string_list(build.get("requires"))]
    tool = data.get("tool") if isinstance(data.get("tool"), dict) else {}
    uv = tool.get("uv") if isinstance(tool.get("uv"), dict) else {}
    found += [("tool.uv.dev-dependencies", r) for r in _string_list(uv.get("dev-dependencies"))]
    sources = uv.get("sources")
    if isinstance(sources, dict):
        found += [("tool.uv.sources", str(name)) for name in sources]
    return found


def _line_naming(text: str, name: str, *, lock: bool = False) -> int | None:
    for number, line in enumerate(text.splitlines(), start=1):
        if lock:
            stripped = line.strip()
            if stripped.startswith("name") and "=" in stripped:
                quoted = _QUOTED_NAME.search(stripped)
                if quoted and normalize_name(quoted.group(1)) == name:
                    return number
            continue
        for quoted in _QUOTED_NAME.finditer(line):
            if name in requirement_names(line[quoted.start() + 1 :]):
                return number
        key = line.split("=", 1)[0].strip()
        if "=" in line and key and normalize_name(key.strip("\"'")) == name:
            return number
    return None


def _load_toml(ws: Workspace, rel: str, result: CheckResult) -> tuple[dict[str, Any], str] | None:
    try:
        raw = ws.read(rel)
    except OSError as error:
        result.add(rel, "unreadable", detail=type(error).__name__)
        return None
    if raw is None:  # tracked, but deleted from the working tree
        result.add(rel, "missing")
        return None
    try:
        text = raw.decode("utf-8")
        return tomllib.loads(text), text
    except UnicodeDecodeError:
        result.add(rel, "not-utf8")
    except tomllib.TOMLDecodeError as error:
        result.add(rel, "toml-parse-error", line=toml_error_line(error))
    return None


def check_deps(root: Path, workspace: Workspace | None = None) -> CheckResult:
    """Fail denied distributions in pyproject.toml and uv.lock (deny list only, see P5)."""
    ws = workspace if workspace is not None else Workspace(root)
    result = CheckResult("deps")
    result.notes.append(P5_NOTE)
    if "pyproject.toml" not in ws.file_set:
        result.add("pyproject.toml", "missing")
    else:
        loaded = _load_toml(ws, "pyproject.toml", result)
        if loaded is not None:
            data, text = loaded
            result.scanned += 1
            for where, requirement in pyproject_requirements(data):
                for name in sorted(requirement_names(requirement) & DENIED_DISTRIBUTIONS):
                    result.add("pyproject.toml", name, line=_line_naming(text, name), detail=where)
    if "uv.lock" in ws.file_set:
        loaded = _load_toml(ws, "uv.lock", result)
        if loaded is not None:
            data, text = loaded
            result.scanned += 1
            packages = data.get("package")
            for package in packages if isinstance(packages, list) else []:
                name = package.get("name") if isinstance(package, dict) else None
                if isinstance(name, str) and normalize_name(name) in DENIED_DISTRIBUTIONS:
                    normalized = normalize_name(name)
                    result.add(
                        "uv.lock", normalized, line=_line_naming(text, normalized, lock=True)
                    )
    else:
        result.notes.append("no uv.lock: only pyproject.toml was checked")
    return result
