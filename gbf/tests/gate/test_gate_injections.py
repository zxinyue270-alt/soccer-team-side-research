"""P0 acceptance for the gate itself (plan P0 and gate G2): the green baseline, the
eight red injections, the restricted material of ADR 0002 section 4 (gate G4), and the
renamed material, reshaped content, dynamic imports and local paths that reviews got past
G3, G4 and the abs_paths check.

Each red test starts from the green baseline repository, injects one problem and
asserts that the gate fails with the expected problem line while printing neither the
injected secret, path or content nor the repository's absolute path.
"""

from __future__ import annotations

import base64
import json
import shutil
import string
from collections.abc import Callable
from pathlib import Path

import numpy as np

from gbf import contracts
from gbf.gate import STATIC_CHECK_NAMES
from gbf.gate.full import run_pytest_check
from gbf.util.hashing import sha256_bytes
from gbf.util.io import write_image_png, write_json, write_text
from gbf.util.paths import repo_root

RunGate = Callable[..., tuple[int, str]]
NoLeak = Callable[..., None]


def _line_of(text: str, needle: str) -> int:
    return next(i for i, line in enumerate(text.splitlines(), start=1) if needle in line)


def test_green_baseline_passes_every_static_check(
    gate_repo: Path, run_gate: RunGate, no_leak: NoLeak
) -> None:
    code, output = run_gate(gate_repo)
    assert code == 0, output
    for name in STATIC_CHECK_NAMES:
        assert f"[gate] {name} ok" in output
    assert "[gate] PASSED" in output
    no_leak(output, gate_repo)


def test_injection_1_denied_dependency(gate_repo: Path, run_gate: RunGate, no_leak: NoLeak) -> None:
    pyproject = gate_repo / "pyproject.toml"
    requirement = "ultralytics>=8.3.40"
    text = pyproject.read_text(encoding="utf-8").replace(
        '"numpy>=2.0",', f'"numpy>=2.0",\n    "{requirement}",'
    )
    write_text(pyproject, text)
    lock = (gate_repo / "uv.lock").read_text(encoding="utf-8")
    lock += '\n[[package]]\nname = "Ultralytics"\nversion = "8.3.40"\n'
    write_text(gate_repo / "uv.lock", lock)

    code, output = run_gate(gate_repo)

    assert code == 1
    line = _line_of(text, requirement)
    assert f"deps pyproject.toml:{line} ultralytics (project.dependencies)" in output
    assert f"deps uv.lock:{_line_of(lock, 'Ultralytics')} ultralytics" in output
    assert "[gate] deps FAILED (2)" in output
    no_leak(output, gate_repo, requirement, "8.3.40")


def test_injection_2_skipped_test_fails_the_pytest_step(
    gate_repo: Path, unicode_tmp_path: Path, git: Callable[..., str], no_leak: NoLeak
) -> None:
    write_json(gate_repo / "gate" / "min_counts.json", {"schema_version": 1, "suites": {}})
    git(gate_repo, "add", "-A")
    green = run_pytest_check(gate_repo, scratch=unicode_tmp_path / "green", counts_base=None)
    assert green.passed, [p.format() for p in green.problems]
    assert green.scanned == 1

    reason = "skip-reason-canary-5d2f"
    write_text(
        gate_repo / "tests" / "test_skipping.py",
        "import pytest\n\n\ndef test_runs() -> None:\n    assert True\n\n\n"
        f"def test_skipped() -> None:\n    pytest.skip({reason!r})\n",
    )
    red = run_pytest_check(gate_repo, scratch=unicode_tmp_path / "red", counts_base=None)

    lines = [p.format() for p in red.problems]
    assert lines == ["pytest tests/test_skipping.py:8 skipped (test_skipped)"]
    assert red.scanned == 3
    no_leak("\n".join(lines + red.notes + red.hints), gate_repo, reason)


def test_injection_3_fake_key_assembled_at_runtime(
    gate_repo: Path, run_gate: RunGate, no_leak: NoLeak, make_token: Callable[..., str]
) -> None:
    key = make_token("s" + "k-", 48)
    write_text(gate_repo / "src" / "demo" / "client.py", f'"""Client."""\n\nOPENAI_KEY = "{key}"\n')

    code, output = run_gate(gate_repo)

    assert code == 1
    assert "credentials src/demo/client.py:3 sk-dash-key" in output
    assert "[gate] credentials FAILED (1)" in output
    no_leak(output, gate_repo, key, key[3:15], key[-12:])


def test_injection_4_unregistered_png(gate_repo: Path, run_gate: RunGate, no_leak: NoLeak) -> None:
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    image[2, 3] = (255, 128, 0)
    write_image_png(gate_repo / "docs" / "golden 帧.png", image)

    code, output = run_gate(gate_repo)

    assert code == 1
    assert "media docs/golden 帧.png unregistered" in output
    assert "[gate] media FAILED (1)" in output
    no_leak(output, gate_repo)


def test_injection_5_user_home_path_assembled_at_runtime(
    gate_repo: Path, run_gate: RunGate, no_leak: NoLeak, make_token: Callable[..., str]
) -> None:
    name = make_token("gbfuser", 6, string.ascii_lowercase)
    windows = chr(92).join(["C:", "Users", name, "Videos", "match.mp4"])
    posix = "/".join(["", "home", name, "clips", "a.mp4"])
    text = f"# Notes\n\nThe clip was cut from {windows}.\nA copy sits in {posix}.\n"
    write_text(gate_repo / "docs" / "notes.md", text)

    code, output = run_gate(gate_repo)

    assert code == 1
    assert "abs_paths docs/notes.md:3 user-home-path" in output
    assert "abs_paths docs/notes.md:4 user-home-path" in output
    no_leak(output, gate_repo, windows, posix, name)


def test_injection_6_receipt_missing_fields(
    gate_repo: Path,
    run_gate: RunGate,
    no_leak: NoLeak,
    valid_receipt: dict[str, object],
    make_token: Callable[..., str],
) -> None:
    name = make_token("gbfuser", 6, string.ascii_lowercase)
    home_path = "/".join(["", "home", name, "inputs", "events.json"])
    producer = "producer-canary-" + make_token("", 10)
    receipt = dict(valid_receipt)
    del receipt["scope"]
    del receipt["limitations"]
    receipt["producer"] = producer
    receipt["inputs"] = [{"name": "events", "relpath": home_path, "sha256": "1" * 64}]
    text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    write_text(gate_repo / "receipts" / "public" / "eval" / "bad.json", text)

    code, output = run_gate(gate_repo)

    assert code == 1
    rel = "receipts/public/eval/bad.json"
    assert f"receipts {rel} schema-required (/scope)" in output
    assert f"receipts {rel} schema-required (/limitations)" in output
    assert f"receipts {rel} schema-not (/inputs/0/relpath)" in output
    assert f"abs_paths {rel}:{_line_of(text, 'relpath')} user-home-path" in output
    no_leak(output, gate_repo, home_path, name, producer)


def test_injection_7_registered_fixture_rewritten_with_crlf(
    gate_repo: Path, run_gate: RunGate, no_leak: NoLeak
) -> None:
    fixture = gate_repo / "tests" / "fixtures" / "sample.txt"
    content = b"alpha\nbeta crlf-canary-line\ngamma\n"
    fixture.parent.mkdir(parents=True)
    fixture.write_bytes(content)
    registry = (
        "schema_version = 1\n\n[[fixtures]]\n"
        'path = "tests/fixtures/sample.txt"\n'
        f'sha256 = "{sha256_bytes(content)}"\n'
    )
    write_text(gate_repo / "registries" / "fixtures.toml", registry)

    _, before = run_gate(gate_repo)
    fixture.write_bytes(content.replace(b"\n", b"\r\n"))
    code_after, after = run_gate(gate_repo)

    eol_problem = "eol tests/fixtures/sample.txt:1 crlf (3 CRLF line endings)"
    hash_problem = (
        "registries tests/fixtures/sample.txt sha256-mismatch "
        "(registered in registries/fixtures.toml)"
    )
    assert eol_problem not in before
    assert hash_problem not in before
    assert "[gate] eol ok" in before
    assert code_after == 1
    assert eol_problem in after
    assert hash_problem in after
    no_leak(after, gate_repo, "crlf-canary-line")


def test_injection_8_forbidden_import(gate_repo: Path, run_gate: RunGate, no_leak: NoLeak) -> None:
    alias = "detector_alias_canary"
    write_text(
        gate_repo / "src" / "demo" / "vision.py",
        f'"""Detector."""\n\nimport ultralytics as {alias}\n',
    )

    code, output = run_gate(gate_repo)

    assert code == 1
    assert "forbidden_imports src/demo/vision.py:3 ultralytics" in output
    assert "[gate] forbidden_imports FAILED (1)" in output
    no_leak(output, gate_repo, alias)


def _problem_lines(output: str, check: str) -> list[str]:
    return [line for line in output.splitlines() if line.startswith(f"{check} ")]


def test_injection_9_restricted_material_by_name(
    gate_repo: Path, run_gate: RunGate, no_leak: NoLeak
) -> None:
    canary = "restricted-name-canary-3b8e"
    injected = {
        "identity_overrides.json": ["identity-overrides"],
        "cases/123.json": ["nda-case-file"],
        "labels/manual_highlights_v1.json": ["manual-labels", "under-labels"],
        "plugin/detections.jsonl": ["raw-plugin-output"],
        "plugin/ball_raw.jsonl": ["raw-plugin-output"],
        "plugin/plugin-receipt.json": ["raw-plugin-output"],
    }
    for rel in injected:
        write_text(gate_repo / rel, json.dumps({"note": canary}) + "\n")

    code, output = run_gate(gate_repo)

    assert code == 1
    expected = [
        f"restricted_data {rel} {rule}" for rel, rules in sorted(injected.items()) for rule in rules
    ]
    assert _problem_lines(output, "restricted_data") == expected
    assert f"[gate] restricted_data FAILED ({len(expected)})" in output
    no_leak(output, gate_repo, canary)


def test_injection_10_restricted_material_by_content(
    gate_repo: Path, run_gate: RunGate, no_leak: NoLeak, valid_receipt: dict[str, object]
) -> None:
    canary = "restricted-content-canary-91d4"
    receipt = dict(valid_receipt, restrictions=["no_share"], producer=canary)
    write_json(gate_repo / "receipts" / "public" / "leak.json", receipt)
    rows = [{"frame": i, "restrictions": ["hl3", "nda"], "note": canary} for i in range(2)]
    write_text(gate_repo / "out" / "rows.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
    write_text(
        gate_repo / "docs" / "report.html",
        f'<!doctype html>\n<div class="private"><strong>私有素材 · 禁止分享</strong>：{canary}'
        "（限制标签：hl3、no_share）</div>\n",
    )

    code, output = run_gate(gate_repo)

    assert code == 1
    assert _problem_lines(output, "restricted_data") == [
        "restricted_data docs/report.html private-banner",
        "restricted_data docs/report.html restricted-tag (no_share)",
        "restricted_data out/rows.jsonl restricted-tag (nda)",
        "restricted_data receipts/public/leak.json restricted-tag (no_share)",
    ]
    assert "[gate] restricted_data FAILED (4)" in output
    no_leak(output, gate_repo, canary)


def test_injection_11_restricted_material_renamed_or_moved(
    gate_repo: Path, run_gate: RunGate, no_leak: NoLeak
) -> None:
    canary = "renamed-material-canary-6c1f"
    injected = {
        "plugin/ball-receipt.json": ["raw-plugin-output"],
        "plugin/sngs_detections.jsonl": ["raw-plugin-output"],
        "plugin/ball-raw.jsonl": ["raw-plugin-output"],
        "calibration-0123456789abcdef/out/tracks.jsonl": ["under-plugin-job"],
        "eval/manual_labels/highlights.json": ["under-labels"],
        "cfg/identity_overrides.yaml": ["identity-overrides"],
        "cases/42.jsonl": ["nda-case-file"],
    }
    for rel in injected:
        write_text(gate_repo / rel, json.dumps({"note": canary}) + "\n")

    code, output = run_gate(gate_repo)

    assert code == 1
    expected = [
        f"restricted_data {rel} {rule}" for rel, rules in sorted(injected.items()) for rule in rules
    ]
    assert _problem_lines(output, "restricted_data") == expected
    assert f"[gate] restricted_data FAILED ({len(expected)})" in output
    no_leak(output, gate_repo, canary)


def test_injection_12_restricted_content_in_other_shapes(
    gate_repo: Path, run_gate: RunGate, no_leak: NoLeak
) -> None:
    canary = "reshaped-content-canary-0a7e"
    write_json(gate_repo / "out" / "run.json", {"run": {"restrictions": ["no_share"]}, "c": canary})
    write_json(gate_repo / "out" / "rows.json", [{"Restrictions": ["nda"], "note": canary}])
    write_text(
        gate_repo / "docs" / "card.svg",
        f"<svg><text>私有<tspan>素材</tspan> {canary}</text></svg>\n",
    )
    write_text(
        gate_repo / "docs" / "table.xhtml",
        f"<table><tr><th>Restrictions</th><td>research_nc</td><td>{canary}</td></tr></table>\n",
    )

    code, output = run_gate(gate_repo)

    assert code == 1
    assert _problem_lines(output, "restricted_data") == [
        "restricted_data docs/card.svg private-banner",
        "restricted_data docs/table.xhtml restricted-tag (research_nc)",
        "restricted_data out/rows.json restricted-tag (nda)",
        "restricted_data out/run.json restricted-tag (no_share)",
    ]
    no_leak(output, gate_repo, canary)


def test_injection_13_forbidden_imports_through_loaders(
    gate_repo: Path, run_gate: RunGate, no_leak: NoLeak
) -> None:
    alias = "loader_alias_canary"
    write_text(
        gate_repo / "src" / "demo" / "vision.py",
        f'"""Detector."""\n\nfrom importlib import import_module as {alias}\n\n'
        f'YOLO = {alias}(name="ultralytics")\n',
    )
    write_text(
        gate_repo / "tests" / "test_vision.py",
        "import importlib.util\n\nimport pytest\n\n"
        'yolo = pytest.importorskip("ultralytics")\n'
        'spec = importlib.util.find_spec("sn_gamestate")\n',
    )

    code, output = run_gate(gate_repo)

    assert code == 1
    assert _problem_lines(output, "forbidden_imports") == [
        "forbidden_imports src/demo/vision.py:5 ultralytics",
        "forbidden_imports tests/test_vision.py:5 ultralytics",
        "forbidden_imports tests/test_vision.py:6 sn_gamestate",
    ]
    no_leak(output, gate_repo, alias)


def test_injection_14_restricted_material_copied_or_restated(
    gate_repo: Path, run_gate: RunGate, no_leak: NoLeak
) -> None:
    # The second review's bypasses, through the CLI: a copy named as a browser names it, a
    # data file named like a JSON Schema, and statements in other shapes and file kinds.
    canary = "restated-material-canary-4e2b"
    write_text(gate_repo / "plugin" / "detections (1).jsonl", json.dumps({"c": canary}) + "\n")
    write_json(gate_repo / "cfg" / "identity_overrides.schema.json", {"note": canary})
    write_text(
        gate_repo / "out" / "rows.ldjson",
        json.dumps({"restrictions": "hl3, nda", "note": canary}) + "\n",
    )
    write_text(
        gate_repo / "docs" / "tags.html",
        f"<p>Restrictions:</p>\n<ul><li>no_share</li></ul>\n<p>{canary}</p>\n",
    )
    write_text(
        gate_repo / "docs" / "card.svg",
        f"<svg><text>限制标签</text><text>research_nc</text><text>{canary}</text></svg>\n",
    )
    write_text(
        gate_repo / "docs" / "report.html.bak", f"<p>私\N{ZERO WIDTH SPACE}有素材 {canary}</p>\n"
    )

    code, output = run_gate(gate_repo)

    assert code == 1
    assert _problem_lines(output, "restricted_data") == [
        "restricted_data cfg/identity_overrides.schema.json identity-overrides",
        "restricted_data docs/card.svg restricted-tag (research_nc)",
        "restricted_data docs/report.html.bak private-banner",
        "restricted_data docs/tags.html restricted-tag (no_share)",
        "restricted_data out/rows.ldjson restricted-tag (nda)",
        "restricted_data plugin/detections (1).jsonl raw-plugin-output",
    ]
    assert "[gate] restricted_data FAILED (6)" in output
    no_leak(output, gate_repo, canary)


def test_injection_15_forbidden_imports_through_other_loaders(
    gate_repo: Path, run_gate: RunGate, no_leak: NoLeak
) -> None:
    # The second review's G3 bypasses, through the CLI: an alias unpacked from a tuple,
    # pkgutil.get_loader, and a .pyw file.
    alias = "unpacked_alias_canary"
    write_text(
        gate_repo / "src" / "demo" / "vision.py",
        f'"""Detector."""\n\nimport importlib\nimport pkgutil\n\n'
        f"{alias}, count = importlib.import_module, 1\n"
        f'YOLO = {alias}("ultralytics")\n'
        'LOADER = pkgutil.get_loader("tracklab")\n',
    )
    write_text(gate_repo / "src" / "demo" / "launcher.pyw", "import sn_gamestate\n")

    code, output = run_gate(gate_repo)

    assert code == 1
    assert _problem_lines(output, "forbidden_imports") == [
        "forbidden_imports src/demo/launcher.pyw:1 sn_gamestate",
        "forbidden_imports src/demo/vision.py:7 ultralytics",
        "forbidden_imports src/demo/vision.py:8 tracklab",
    ]
    no_leak(output, gate_repo, alias)


def test_injection_16_restricted_content_the_third_review_restated(
    gate_repo: Path, run_gate: RunGate, no_leak: NoLeak
) -> None:
    # The third review's G4 bypasses, through the CLI: a string after a quoted key in a file
    # with comments, a label word split by <wbr>, and a saved page nested too deeply for
    # Python's email parser, which failed the whole check as an internal error that named
    # no file and hid the other two.
    canary = "third-review-canary-58c1"
    write_text(
        gate_repo / "out" / "export.jsonc", f'{{\n  // {canary}\n  "restrictions": "no_share"\n}}\n'
    )
    write_text(
        gate_repo / "docs" / "report.html", f"<p>Re<wbr>strictions: nda</p><p>{canary}</p>\n"
    )
    nested = "Content-Type: message/rfc822\n\n" * 1000
    write_text(
        gate_repo / "docs" / "saved.mhtml",
        f"MIME-Version: 1.0\n{nested}Content-Type: text/html\n\n<p>{canary}</p>\n",
    )

    code, output = run_gate(gate_repo)

    assert code == 1
    assert _problem_lines(output, "restricted_data") == [
        "restricted_data docs/report.html restricted-tag (nda)",
        "restricted_data docs/saved.mhtml mime-parse-error (parts nested too deeply)",
        "restricted_data out/export.jsonc restricted-tag (no_share)",
    ]
    assert "[gate] restricted_data FAILED (3)" in output
    no_leak(output, gate_repo, canary)


def test_injection_17_restricted_content_the_fourth_review_restated(
    gate_repo: Path, run_gate: RunGate, no_leak: NoLeak
) -> None:
    # The fourth review's G4 bypasses, through the CLI: an apostrophe in the string after a
    # quoted key, a copy of hand labels under another name, and a saved page whose charset
    # holds a NUL, which failed the whole check as an internal error that named no file and
    # hid the other two.
    canary = "fourth-review-canary-7d3a"
    write_text(
        gate_repo / "out" / "export.jsonc",
        f'{{\n  // {canary}\n  "restrictions": "partner\'s nda"\n}}\n',
    )
    moment = {"id": "M1", "type": "duel_tackle", "start": 100, "peak": 110, "end": 120}
    write_json(gate_repo / "notes" / "highlights_v1.json", {"id": canary, "moments": [moment]})
    charset = "ut" + chr(0) + "f-8"
    write_text(
        gate_repo / "docs" / "saved.mhtml",
        f'MIME-Version: 1.0\nContent-Type: text/html; charset="{charset}"\n\n<p>{canary}</p>\n',
    )

    code, output = run_gate(gate_repo)

    assert code == 1
    assert _problem_lines(output, "restricted_data") == [
        "restricted_data docs/saved.mhtml mime-parse-error (unusable charset)",
        "restricted_data notes/highlights_v1.json manual-labels-content",
        "restricted_data out/export.jsonc restricted-tag (nda)",
    ]
    assert "[gate] restricted_data FAILED (3)" in output
    no_leak(output, gate_repo, canary)


def test_injection_18_paths_and_restricted_content_the_fifth_review_restated(
    gate_repo: Path, run_gate: RunGate, no_leak: NoLeak
) -> None:
    # The fifth review's bypasses, through the CLI: a private-data path that steps into the
    # data root through a .. part, an attribute value that ends in a backslash, and a saved
    # page whose charset is an RFC 2231 parameter written in a charset with a NUL, which
    # failed the whole check as an internal error that named no file and hid the other.
    canary = "fifth-review-canary-2b9e"
    path = "/".join(["D:", "work", "gbf", "..", "gbf" + "-data", "priv" + "ate", canary])
    write_text(gate_repo / "docs" / "notes.md", f"# Notes\n\nRun written to {path}.\n")
    write_text(gate_repo / "docs" / "card.html", f'<div data-restrictions="nda\\">{canary}</div>\n')
    part = f"<p>Restrictions: no_share</p><p>{canary}</p>\n"
    encoded = base64.b64encode(part.encode("utf-8")).decode("ascii")
    charset = "ut" + chr(0) + "f-8"
    write_text(
        gate_repo / "docs" / "saved.mhtml",
        f"MIME-Version: 1.0\nContent-Type: text/html; charset*={charset}''utf-8\n"
        f"Content-Transfer-Encoding: base64\n\n{encoded}\n",
    )

    code, output = run_gate(gate_repo)

    assert code == 1
    assert _problem_lines(output, "abs_paths") == ["abs_paths docs/notes.md:3 private-data-path"]
    assert _problem_lines(output, "restricted_data") == [
        "restricted_data docs/card.html restricted-tag (nda)",
        "restricted_data docs/saved.mhtml mime-parse-error (unusable charset)",
        "restricted_data docs/saved.mhtml restricted-tag (no_share)",
    ]
    assert "[gate] restricted_data FAILED (3)" in output
    no_leak(output, gate_repo, canary, path)


def test_schemas_registries_and_public_material_pass_the_restricted_data_check(
    gate_repo: Path, run_gate: RunGate, valid_receipt: dict[str, object]
) -> None:
    # The project's own JSON Schemas list every restriction tag in enums, and its
    # registries assign tags to datasets and models: neither is restricted material.
    shutil.copytree(contracts.SCHEMA_DIR, gate_repo / "src" / "gbf" / "contracts" / "schemas")
    shutil.copytree(repo_root() / "registries", gate_repo / "registries")
    public = dict(valid_receipt, restrictions=["derived_aggregate", "non_commercial"])
    write_json(gate_repo / "receipts" / "public" / "aggregate.json", public)
    write_json(gate_repo / "docs" / "nested.json", {"items": [{"restrictions": ["no_share"]}]})
    write_text(gate_repo / "docs" / "rows.jsonl", json.dumps({"restrictions": []}) + "\n")
    write_text(
        gate_repo / "docs" / "index.html",
        "<!doctype html>\n<p>Restriction tags such as no_share and nda are listed in ADR "
        '0002.</p>\n<pre>{"restrictions": []}</pre>\n',
    )
    write_text(gate_repo / "docs" / "policy.md", "Reports show the banner 私有素材.\n")
    # Code, docs and schemas named like restricted material are not restricted material.
    schema = (contracts.SCHEMA_DIR / "plugin-receipt.schema.json").read_text(encoding="utf-8")
    write_text(gate_repo / "src" / "demo" / "schemas" / "detections.schema.json", schema)
    write_text(gate_repo / "src" / "demo" / "identity_overrides.py", '"""Overrides."""\n')
    write_text(gate_repo / "src" / "demo" / "ball_raw.py", '"""Ball candidates."""\n')
    write_text(gate_repo / "docs" / "detections.md", "# The detections.jsonl format\n")
    write_text(
        gate_repo / "docs" / "tags.html",
        "<h2>Restrictions:</h2>\n<p>Tags such as no_share and nda travel with every artifact."
        "</p>\n<p>Restrictions: none.</p>\n",
    )

    code, output = run_gate(gate_repo)

    assert code == 0, output
    assert "[gate] restricted_data ok" in output
    assert "[gate] registries ok" in output
