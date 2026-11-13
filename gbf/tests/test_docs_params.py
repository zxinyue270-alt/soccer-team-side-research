"""docs/design/m1-highlights.md, section "Tuned parameters", agrees with the code.

ADR 0002 (amendment of 2026-10-21) has every tuned parameter of the M1 pipeline recorded in
that section with its value, kind and provenance. These tests parse its tables (the format
is described in the section's first paragraphs) and check:

- every listed name exists and has the listed value in the code: a module constant, a
  dataclass field default, an attribute of a module object, a mapping entry (for the
  kinematics settings the pipeline passes, ``gbf.pipeline.KINEMATICS_PARAMS``) or a
  parameter default; every literal row's code is still in its function;
- every numeric constant of an M1 module, every numeric field default of a listed
  dataclass, every field a listed object changes and every entry of a listed mapping is
  either listed or named in :data:`NOT_TUNED` with the reason it is not a tuned parameter.

A failure names the doc line or the parameter. Fix the doc or the code and say how the
value was chosen; never loosen a comparison here to make a row pass.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import math
import numbers
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from gbf import ball, pipeline, team_check
from gbf.ingest import cuts
from gbf.util.paths import repo_root

DOC = repo_root(Path(__file__)) / "docs" / "design" / "m1-highlights.md"
SECTION = "## Tuned parameters"
COLUMNS = ("Parameter", "Value", "Kind", "Provenance")
KINDS = ("fitted", "hand-set", "football fact")

M1_MODULES = (
    "gbf.sngs_import",
    "gbf.ingest.cuts",
    "gbf.shots_grammar",
    "gbf.enhance",
    "gbf.track_tools",
    "gbf.calib_refine",
    "gbf.team_check",
    "gbf.role_check",
    "gbf.side_check",
    "gbf.kinematics",
    "gbf.ball",
    "gbf.audio",
    "gbf.highlights",
    "gbf.pose",
    "gbf.lift3d",
    "gbf.moments",
    "gbf.measure",
    "gbf.render",
    "gbf.report",
    "gbf.label_eval",
    "gbf.pipeline",
)
"""The modules of the M1 pipeline (gbf.pipeline's stages and the modules they call)."""

NO_TABLE_MODULES = ("gbf.enhance", "gbf.pose", "gbf.report", "gbf.label_eval", "gbf.pipeline")
"""M1 modules without a table of their own: no tuned parameter of their own, or (gbf.pipeline)
their settings are listed in the table of the module they are passed to."""

NOT_TUNED: Mapping[str, Mapping[str, str]] = {
    "gbf.sngs_import": {"MAX_LISTED_PROBLEMS": "length of an error report"},
    "gbf.shots_grammar": {"REFERENCE_HEIGHT_PX": "the 1080-line reference height"},
    "gbf.track_tools": {"ID_STRIDE": "id scheme of tracks split at cuts"},
    "gbf.calib_refine": {
        "_LOGF_MIN": "focal-length bound of the fit (numerical guard)",
        "_LOGF_MAX": "focal-length bound of the fit (numerical guard)",
    },
    "gbf.kinematics": {
        "_KEPT": "code of a kept position",
        "_BOX_HEIGHT": "code of a rejection test",
        "_BOX_WIDTH": "code of a rejection test",
        "_LOCAL_MEDIAN": "code of a rejection test",
        "_EXCURSION": "code of a rejection test",
    },
    "gbf.ball": {
        "BallParams.reference_lines": "the 1080-line reference height",
        "BallParams.conf_floor": "numerical guard before a logarithm",
    },
    "gbf.audio": {
        "RMS_FLOOR_DBFS": "floor for digital silence (numerical guard)",
        "BLOCK_FRAMES": "memory block size; no effect on results",
        "MIN_SAMPLE_RATE": "input check",
        "MAX_SAMPLE_RATE": "input check",
    },
    "gbf.lift3d": {
        "_HIP_PRIOR": "entry of JOINT_HEIGHT_PRIOR_M",
        "_SHOULDER_PRIOR": "entry of JOINT_HEIGHT_PRIOR_M",
    },
    "gbf.measure": {
        "SCHEMA_VERSION": "record format",
        "FLOAT_DECIMALS": "record format (rounding before hashing)",
        "REFERENCE_HEIGHT_PX": "the 1080-line reference height",
        "CARD_LABEL_MAX_CHARS": "data card layout",
    },
    "gbf.render": {
        "RING_RX_PER_BOX_WIDTH": "drawing: the ground ellipse under a player",
        "RING_RY_PER_RX": "drawing: the ground ellipse under a player",
        "FACING_ARROW_LENGTH_M": "drawing: the length of the facing arrow on the grass",
        "AUDIO_SAMPLE_RATE": "output format",
        "AUDIO_CHANNELS": "output format",
        "DEFAULT_CRF": "encoder setting",
        "X264_THREADS": "encoder setting",
    },
    "gbf.pose": {
        "N_KEYPOINTS": "Halpe-26 format",
        "UNKNOWN_TRACK_ID": "id of a box without a track",
    },
    "gbf.label_eval": {"DEFAULT_TOLERANCE_FRAMES": "evaluation setting"},
    "gbf.pipeline": {
        "CARD_MAX_ROWS": "data card layout",
        "CARD_MOMENT_ROWS": "data card layout",
        "POSE_RANGE_MERGE_GAP": "decode ranges of the pose stage; no effect on results",
        "PipelineConfig.audio_offset_s": "input alignment, set per run",
    },
}
"""Numeric names of the M1 modules that are not tuned parameters, with the reason."""

_HEADING = re.compile(r"### (?P<module>gbf(?:\.[a-z_][a-z0-9_]*)+)\b")
_NAMES_CELL = re.compile(r"`[^`]+`(?:, `[^`]+`)*")
_NAME = re.compile(r"`([^`]+)`")
_LITERAL_CELL = re.compile(r"literal in `(?P<function>[A-Za-z_][\w.]*)`: `(?P<code>[^`]+)`")
_SEPARATOR = re.compile(r":?-{3,}:?")
_MEMBER = re.compile(
    r"(?P<owner>[A-Za-z_]\w*)"
    r"(?:\.(?P<field>[A-Za-z_]\w*)|\[(?P<key>[A-Za-z_]\w*)\]|\((?P<param>[A-Za-z_]\w*)\))?"
)
_CONSTANT = re.compile(r"_?[A-Z][A-Z0-9_]*")


@dataclass(frozen=True)
class Row:
    """One table row: ``names`` and ``values`` as written (a literal row has ``function``
    and ``code`` instead of names)."""

    line: int
    module: str
    names: tuple[str, ...]
    values: tuple[str, ...]
    kind: str
    provenance: str
    function: str | None = None
    code: str | None = None


@dataclass(frozen=True)
class Ref:
    """A listed name resolved in the code. ``form`` is name, field (dataclass field
    default), attr (attribute of a module object), item (mapping entry) or param."""

    module: str
    owner: str
    form: str
    member: str | None
    value: Any


# --------------------------------------------------------------------------- parsing


def _cells(line: str) -> list[str]:
    body = line.strip()
    body = body[1:] if body.startswith("|") else body
    body = body[:-1] if body.endswith("|") else body
    return [cell.strip() for cell in body.split("|")]


def _row(number: int, module: str, cells: list[str], problems: list[str]) -> Row | None:
    if len(cells) != len(COLUMNS):
        problems.append(f"line {number}: expected {len(COLUMNS)} cells, got {len(cells)}")
        return None
    parameter, value, kind, provenance = cells
    found: list[str] = []
    if kind not in KINDS:
        found.append(f"kind must be one of {', '.join(KINDS)}, got {kind!r}")
    if not provenance:
        found.append("the provenance is empty")
    values = tuple(item.strip() for item in value.split(";"))
    if not all(values):
        found.append(f"empty value in {value!r}")
    names: tuple[str, ...] = ()
    function = code = None
    literal = _LITERAL_CELL.fullmatch(parameter)
    if literal is not None:
        function, code = literal["function"], literal["code"]
        if len(values) != 1:
            found.append("a literal row has exactly one value")
    elif _NAMES_CELL.fullmatch(parameter):
        names = tuple(_NAME.findall(parameter))
        if len(values) not in (1, len(names)):
            found.append(f"{len(names)} names but {len(values)} values")
    else:
        found.append(
            "the parameter cell must hold names in backticks separated by ', ', "
            "or read: literal in `function`: `code`"
        )
    if found:
        problems.extend(f"line {number}: {problem}" for problem in found)
        return None
    return Row(number, module, names, values, kind, provenance, function, code)


def parse_section(text: str) -> tuple[list[Row], list[str], list[str]]:
    """(rows, module of each table heading in order, format problems) of the section."""
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == SECTION]
    if len(starts) != 1:
        return [], [], [f"expected one {SECTION!r} heading, found {len(starts)}"]
    first = starts[0] + 1
    last = next((i for i in range(first, len(lines)) if lines[i].startswith("## ")), len(lines))
    rows: list[Row] = []
    modules: list[str] = []
    problems: list[str] = []
    module: str | None = None
    state = "text"
    for index in range(first, last):
        line, number = lines[index], index + 1
        if line.startswith("### "):
            heading = _HEADING.match(line)
            if heading is None:
                problems.append(f"line {number}: a heading must start with a gbf module name")
                module = None
            else:
                module = heading["module"]
                if module in modules:
                    problems.append(f"line {number}: a second table heading for {module}")
                modules.append(module)
            state = "text"
            continue
        if not line.startswith("|"):
            if state == "separator":
                problems.append(f"line {number - 1}: a table header without a separator row")
            state = "text"
            continue
        cells = _cells(line)
        if state == "text":
            if module is None:
                problems.append(f"line {number}: a table outside a module's heading")
            if tuple(cells) != COLUMNS:
                problems.append(f"line {number}: the header must read {' | '.join(COLUMNS)}")
            state = "separator"
        elif state == "separator":
            if len(cells) != len(COLUMNS) or not all(_SEPARATOR.fullmatch(c) for c in cells):
                problems.append(f"line {number}: expected a separator row |---|---|---|---|")
            state = "rows"
        else:
            row = _row(number, module or "", cells, problems)
            if row is not None:
                rows.append(row)
    return rows, modules, problems


# --------------------------------------------------------------------------- the code


def _importable(name: str) -> bool:
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name is not None and (name == exc.name or name.startswith(exc.name + ".")):
            return False
        raise
    return True


def _split(name: str, default: str) -> tuple[str, str]:
    """(module, member) of a listed name; ``gbf.x.y.MEMBER`` names its own module."""
    if not name.startswith("gbf."):
        return default, name
    parts = re.split(r"[\[(]", name, maxsplit=1)[0].split(".")
    for end in range(len(parts) - 1, 1, -1):
        candidate = ".".join(parts[:end])
        if _importable(candidate):
            return candidate, name[len(candidate) + 1 :]
    raise LookupError(f"no gbf module in {name!r}")


def _field_default(field: dataclasses.Field[Any]) -> Any:
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:
        return field.default_factory()
    return dataclasses.MISSING


def resolve(name: str, default_module: str) -> Ref:
    """What ``name`` denotes in the code (LookupError when it denotes nothing)."""
    module_name, member = _split(name, default_module)
    module = importlib.import_module(module_name)
    match = _MEMBER.fullmatch(member)
    if match is None:
        raise LookupError(
            "expected NAME, Class.field, OBJECT.field, NAME[key] or function(parameter)"
        )
    owner_name = match["owner"]
    if not hasattr(module, owner_name):
        raise LookupError(f"{module_name} has no {owner_name}")
    owner = getattr(module, owner_name)
    if match["field"] is not None:
        field_name = match["field"]
        if isinstance(owner, type) and dataclasses.is_dataclass(owner):
            fields = {item.name: item for item in dataclasses.fields(owner)}
            if field_name not in fields:
                raise LookupError(f"{module_name}.{owner_name} has no field {field_name}")
            value = _field_default(fields[field_name])
            if value is dataclasses.MISSING:
                raise LookupError(f"{module_name}.{owner_name}.{field_name} has no default")
            return Ref(module_name, owner_name, "field", field_name, value)
        if not hasattr(owner, field_name):
            raise LookupError(f"{module_name}.{owner_name} has no attribute {field_name}")
        return Ref(module_name, owner_name, "attr", field_name, getattr(owner, field_name))
    if match["key"] is not None:
        if not isinstance(owner, Mapping) or match["key"] not in owner:
            raise LookupError(f"{module_name}.{owner_name} has no entry {match['key']!r}")
        return Ref(module_name, owner_name, "item", match["key"], owner[match["key"]])
    if match["param"] is not None:
        if not callable(owner):
            raise LookupError(f"{module_name}.{owner_name} is not a function")
        parameter = inspect.signature(owner).parameters.get(match["param"])
        if parameter is None or parameter.default is inspect.Parameter.empty:
            raise LookupError(f"{module_name}.{owner_name} has no default for {match['param']}")
        return Ref(module_name, owner_name, "param", match["param"], parameter.default)
    return Ref(module_name, owner_name, "name", None, owner)


def same(expected: Any, actual: Any) -> bool:
    """The doc's literal and the code's value are the same parameter value."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return isinstance(expected, bool) and isinstance(actual, bool) and expected == actual
    if expected is None or actual is None:
        return expected is None and actual is None
    if isinstance(expected, numbers.Real) and isinstance(actual, numbers.Real):
        return math.isclose(float(expected), float(actual), rel_tol=1e-12, abs_tol=1e-15)
    if isinstance(expected, str) or isinstance(actual, str):
        return isinstance(expected, str) and isinstance(actual, str) and expected == actual
    if isinstance(expected, tuple) and isinstance(actual, tuple | list):
        return len(expected) == len(actual) and all(map(same, expected, actual))
    return False


def _squeezed(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _literal_problems(row: Row) -> list[str]:
    where = f"line {row.line}: literal in {row.function}"
    value = row.values[0]
    if re.search(rf"(?<![\w.]){re.escape(value)}(?![\w.])", row.code or "") is None:
        return [f"{where}: the value {value} is not in the quoted code {row.code!r}"]
    try:
        target: Any = importlib.import_module(row.module)
        for part in (row.function or "").split("."):
            target = getattr(target, part)
        source = inspect.getsource(target)
    except (AttributeError, OSError, TypeError) as exc:
        return [f"{where}: no such function in {row.module} ({exc})"]
    if _squeezed(row.code or "") not in _squeezed(source):
        return [f"{where}: the function no longer contains {row.code!r}"]
    return []


def check_rows(rows: Iterable[Row]) -> tuple[list[Ref], list[str]]:
    """(the resolved names, problems): names that denote nothing, values that differ,
    names listed twice and literal rows whose code is gone."""
    refs: list[Ref] = []
    problems: list[str] = []
    seen: dict[tuple[str, str, str, str | None], int] = {}
    for row in rows:
        if row.function is not None:
            problems.extend(_literal_problems(row))
            continue
        texts = row.values * len(row.names) if len(row.values) == 1 else row.values
        for name, text in zip(row.names, texts, strict=True):
            try:
                expected = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                problems.append(f"line {row.line}: {name}: {text!r} is not a Python literal")
                continue
            try:
                ref = resolve(name, row.module)
            except LookupError as exc:
                problems.append(f"line {row.line}: {name}: {exc}")
                continue
            key = (ref.module, ref.owner, ref.form, ref.member)
            if key in seen:
                problems.append(f"line {row.line}: {name} is listed again (line {seen[key]})")
            seen[key] = row.line
            refs.append(ref)
            if not same(expected, ref.value):
                problems.append(
                    f"line {row.line}: {name}: the doc says {text}, the code has {ref.value!r}"
                )
    return refs, problems


def _is_number(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _numeric(value: Any) -> bool:
    if isinstance(value, tuple):
        return bool(value) and all(_is_number(item) for item in value)
    return _is_number(value)


def module_constants(module: ModuleType) -> dict[str, Any]:
    """Numeric scalar constants (UPPER_CASE, private ones too) the module assigns itself."""
    source = Path(inspect.getsourcefile(module) or "").read_text(encoding="utf-8")
    names: list[str] = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
        else:
            continue
        for target in targets:
            items = target.elts if isinstance(target, ast.Tuple) else [target]
            names.extend(item.id for item in items if isinstance(item, ast.Name))
    return {
        name: getattr(module, name)
        for name in names
        if _CONSTANT.fullmatch(name) and _is_number(getattr(module, name, None))
    }


def unlisted(refs: Iterable[Ref], modules: Iterable[str]) -> list[str]:
    """Numeric parameters that are neither listed nor in :data:`NOT_TUNED`."""
    refs = list(refs)
    listed = {(r.module, r.owner, r.form, r.member) for r in refs}
    found: list[str] = []

    def skipped(module_name: str, name: str) -> bool:
        return name in NOT_TUNED.get(module_name, {})

    for module_name in modules:
        module = importlib.import_module(module_name)
        for name, value in module_constants(module).items():
            if (module_name, name, "name", None) not in listed and not skipped(module_name, name):
                found.append(f"{module_name}.{name} = {value!r}")
    for module_name, owner, form in sorted({(r.module, r.owner, r.form) for r in refs}):
        holder = getattr(importlib.import_module(module_name), owner)
        if form == "field":
            for field in dataclasses.fields(holder):
                default = _field_default(field)
                name = f"{owner}.{field.name}"
                if default is dataclasses.MISSING or not _numeric(default):
                    continue
                if (module_name, owner, form, field.name) not in listed and not skipped(
                    module_name, name
                ):
                    found.append(f"{module_name}.{name} = {default!r}")
        elif form == "attr" and dataclasses.is_dataclass(holder) and not isinstance(holder, type):
            for field in dataclasses.fields(holder):
                value = getattr(holder, field.name)
                name = f"{owner}.{field.name}"
                if value == _field_default(field) or not _numeric(value):
                    continue
                if (module_name, owner, form, field.name) not in listed and not skipped(
                    module_name, name
                ):
                    found.append(f"{module_name}.{name} = {value!r} (differs from its class)")
        elif form == "item":
            for key in holder:
                name = f"{owner}[{key}]"
                if (module_name, owner, form, key) not in listed and not skipped(module_name, name):
                    found.append(f"{module_name}.{name} = {holder[key]!r}")
    return found


# --------------------------------------------------------------------------- the doc


def _doc() -> tuple[list[Row], list[str], list[str]]:
    return parse_section(DOC.read_text(encoding="utf-8"))


def test_the_section_follows_its_table_format() -> None:
    rows, modules, problems = _doc()
    assert problems == [], "\n".join(problems)
    assert rows, f"no table rows under {SECTION!r} in {DOC.name}"
    unknown = sorted(set(modules) - set(M1_MODULES))
    assert unknown == [], f"tables for modules outside M1_MODULES: {unknown}"
    both = sorted(set(modules) & set(NO_TABLE_MODULES))
    assert both == [], f"modules with a table that NO_TABLE_MODULES lists: {both}"
    missing = sorted(set(M1_MODULES) - set(modules) - set(NO_TABLE_MODULES))
    assert missing == [], f"M1 modules without a table: {missing}"


def test_every_listed_value_is_the_code_value() -> None:
    rows, _modules, problems = _doc()
    assert problems == [], "\n".join(problems)
    _refs, found = check_rows(rows)
    assert found == [], f"{DOC.name}, {SECTION!r}:\n" + "\n".join(found)


def test_every_numeric_parameter_is_listed_or_not_tuned() -> None:
    rows, _modules, problems = _doc()
    assert problems == [], "\n".join(problems)
    refs, _found = check_rows(rows)
    found = unlisted(refs, M1_MODULES)
    assert found == [], (
        f"add a row to {DOC.name} ({SECTION!r}), or name it in NOT_TUNED with the reason it "
        "is no tuned parameter:\n" + "\n".join(found)
    )


def test_not_tuned_names_exist_and_are_not_listed() -> None:
    rows, _modules, problems = _doc()
    assert problems == [], "\n".join(problems)
    refs, _found = check_rows(rows)
    listed = {(r.module, r.owner, r.form, r.member) for r in refs}
    found = []
    for module_name, names in NOT_TUNED.items():
        if module_name not in M1_MODULES:
            found.append(f"{module_name} is not in M1_MODULES")
            continue
        constants = module_constants(importlib.import_module(module_name))
        for name, reason in names.items():
            try:
                ref = resolve(name, module_name)
            except LookupError as exc:
                found.append(f"{module_name}: {name}: {exc}")
                continue
            if not reason:
                found.append(f"{module_name}: {name} needs a reason")
            if ref.form == "name" and name not in constants:
                found.append(f"{module_name}: {name} is no numeric constant the module assigns")
            elif ref.form != "name" and not _numeric(ref.value):
                found.append(f"{module_name}: {name} is not numeric")
            if (ref.module, ref.owner, ref.form, ref.member) in listed:
                found.append(f"{module_name}: {name} is in NOT_TUNED and in the doc")
    assert found == [], "\n".join(found)


# --------------------------------------------------------------------------- the check
# These tests run the check on samples whose expected results are read from the code, so
# they keep passing when a parameter changes and the doc follows it.


def _sample() -> str:
    window = pipeline.KINEMATICS_PARAMS["sg_window_frames"]
    bins, sample = cuts.BINS_PER_CHANNEL, cuts.SAMPLE_ROWS
    return f"""\
# Sample

## Tuned parameters

Text before the tables is ignored.

### gbf.ingest.cuts (hard cuts)

| Parameter | Value | Kind | Provenance |
|---|---|---|---|
| `DEFAULT_CUT_THRESHOLD` | {cuts.DEFAULT_CUT_THRESHOLD + 0.125!r} | hand-set | a wrong value |
| `DEFAULT_CUT_THRESHOLDS` | 0.5 | hand-set | a wrong name |
| `BINS_PER_CHANNEL`, `SAMPLE_ROWS` | {bins!r}; {sample!r} | hand-set | right |
| `gbf.pipeline.KINEMATICS_PARAMS[sg_window_frames]` | {window!r} | hand-set | another module |
| literal in `detect_cuts`: `score > limit` | 0.5 | hand-set | parsed only |

## Next section

| `NOT_PARSED` | 1 | hand-set | outside the section |
"""


def _threshold_example(score: float) -> bool:
    """A function with a literal threshold, for the literal-row checks below."""
    return score >= 0.25


def test_the_check_reports_a_wrong_value_and_a_wrong_name() -> None:
    rows, modules, problems = parse_section(_sample())
    assert problems == []
    assert modules == ["gbf.ingest.cuts"]
    assert [row.line for row in rows] == [11, 12, 13, 14, 15]
    assert (rows[4].names, rows[4].function, rows[4].code) == ((), "detect_cuts", "score > limit")
    refs, found = check_rows(rows[:4])
    assert [(r.module, r.owner, r.form, r.member, r.value) for r in refs] == [
        ("gbf.ingest.cuts", "DEFAULT_CUT_THRESHOLD", "name", None, cuts.DEFAULT_CUT_THRESHOLD),
        ("gbf.ingest.cuts", "BINS_PER_CHANNEL", "name", None, cuts.BINS_PER_CHANNEL),
        ("gbf.ingest.cuts", "SAMPLE_ROWS", "name", None, cuts.SAMPLE_ROWS),
        (
            "gbf.pipeline",
            "KINEMATICS_PARAMS",
            "item",
            "sg_window_frames",
            pipeline.KINEMATICS_PARAMS["sg_window_frames"],
        ),
    ]
    wrong = rows[0].values[0]
    assert found == [
        f"line 11: DEFAULT_CUT_THRESHOLD: the doc says {wrong}, "
        f"the code has {cuts.DEFAULT_CUT_THRESHOLD!r}",
        "line 12: DEFAULT_CUT_THRESHOLDS: gbf.ingest.cuts has no DEFAULT_CUT_THRESHOLDS",
    ]
    _refs, found = check_rows([rows[2], rows[2]])
    assert found == [
        "line 13: BINS_PER_CHANNEL is listed again (line 13)",
        "line 13: SAMPLE_ROWS is listed again (line 13)",
    ]


def test_the_check_reports_stale_literal_rows() -> None:
    def row(value: str, code: str, function: str = "_threshold_example") -> Row:
        return Row(7, __name__, (), (value,), "hand-set", "sample", function, code)

    assert _threshold_example(0.25)
    assert check_rows([row("0.25", "score >= 0.25")]) == ([], [])
    assert check_rows([row("0.5", "score >= 0.25")])[1] == [
        "line 7: literal in _threshold_example: the value 0.5 is not in the quoted code "
        "'score >= 0.25'"
    ]
    assert check_rows([row("0.3", "score >= 0.3")])[1] == [
        "line 7: literal in _threshold_example: the function no longer contains 'score >= 0.3'"
    ]
    found = check_rows([row("1", "x = 1", "_no_such_function")])[1]
    assert len(found) == 1
    assert found[0].startswith("line 7: literal in _no_such_function: no such function")


def test_the_check_reports_format_errors() -> None:
    bad = (
        _sample()
        .replace(
            "| `DEFAULT_CUT_THRESHOLDS` | 0.5 | hand-set | a wrong name |",
            "| `DEFAULT_CUT_THRESHOLDS`, `SAMPLE_ROWS` | 1; 2; 3 | tuned | |\n"
            "| DEFAULT_CUT_THRESHOLD | 0.5 | hand-set | no backticks |\n"
            "| `SAMPLE_ROWS` | 72 | hand-set |",
        )
        .replace("|---|---|---|---|\n", "")
    )
    rows, _modules, problems = parse_section(bad)
    assert problems == [
        "line 10: expected a separator row |---|---|---|---|",
        f"line 11: kind must be one of {', '.join(KINDS)}, got 'tuned'",
        "line 11: the provenance is empty",
        "line 11: 2 names but 3 values",
        "line 12: the parameter cell must hold names in backticks separated by ', ', "
        "or read: literal in `function`: `code`",
        "line 13: expected 4 cells, got 3",
    ]
    assert [row.line for row in rows] == [14, 15, 16]
    headings = _sample().replace("### gbf.ingest.cuts (hard cuts)", "### Hard cuts")
    assert parse_section(headings)[2][0] == "line 7: a heading must start with a gbf module name"
    missing = _sample().replace("## Tuned parameters", "## Tuned")
    assert parse_section(missing)[2] == [f"expected one {SECTION!r} heading, found 0"]


def test_the_check_reports_unlisted_parameters() -> None:
    constants = sorted(module_constants(cuts))
    listed = [resolve(name, "gbf.ingest.cuts") for name in constants]
    assert unlisted(listed, ["gbf.ingest.cuts"]) == []
    dropped = [name for name in constants if name not in NOT_TUNED.get("gbf.ingest.cuts", {})][-1]
    assert unlisted([r for r in listed if r.owner != dropped], ["gbf.ingest.cuts"]) == [
        f"gbf.ingest.cuts.{dropped} = {getattr(cuts, dropped)!r}"
    ]

    key = next(iter(pipeline.KINEMATICS_PARAMS))
    found = unlisted([resolve(f"gbf.pipeline.KINEMATICS_PARAMS[{key}]", "gbf.ball")], [])
    assert len(found) == len(pipeline.KINEMATICS_PARAMS) - 1
    assert not any(f"[{key}] =" in line for line in found)

    defaults = {
        f.name: _field_default(f)
        for f in dataclasses.fields(team_check.TeamCheckParams)
        if _numeric(_field_default(f))
        and f"TeamCheckParams.{f.name}" not in NOT_TUNED.get("gbf.team_check", {})
    }
    first, *others = defaults
    found = unlisted([resolve(f"gbf.team_check.TeamCheckParams.{first}", "gbf.ball")], [])
    assert others
    assert found == [
        f"gbf.team_check.TeamCheckParams.{name} = {defaults[name]!r}" for name in others
    ]

    differing = [
        f.name
        for f in dataclasses.fields(ball.CONTACT_PARAMS)
        if getattr(ball.CONTACT_PARAMS, f.name) != getattr(ball.DEFAULT_PARAMS, f.name)
    ]
    assert differing
    found = unlisted([resolve(f"CONTACT_PARAMS.{differing[0]}", "gbf.ball")], [])
    assert found == [
        f"gbf.ball.CONTACT_PARAMS.{name} = {getattr(ball.CONTACT_PARAMS, name)!r} "
        "(differs from its class)"
        for name in differing[1:]
    ]
