"""Each static check on its own, run against the temporary baseline repository."""

from __future__ import annotations

import base64
import codecs
import json
import quopri
import re
import shutil
import string
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from gbf import label_eval
from gbf.gate import datafiles as datafiles_module
from gbf.gate import imports as imports_module
from gbf.gate import repo as repo_module
from gbf.gate.content import check_abs_paths, check_credentials, check_eol
from gbf.gate.datafiles import check_media, check_restricted_data
from gbf.gate.imports import check_deps, check_forbidden_imports, requirement_names
from gbf.gate.model import CheckResult, GateError
from gbf.gate.registries import (
    check_receipts,
    check_registries,
    registry_entries,
    schema_problems,
)
from gbf.gate.repo import Workspace, candidate_files, to_json_compatible
from gbf.models import PINS
from gbf.util.hashing import sha256_bytes, sha256_file
from gbf.util.io import write_image_png, write_json, write_text
from gbf.util.paths import repo_root


def lines(result: CheckResult) -> list[str]:
    return [p.format() for p in result.sorted_problems()]


def _line_of(text: str, needle: str) -> int:
    return next(i for i, line in enumerate(text.splitlines(), start=1) if needle in line)


def _png(path: Path, value: int = 0) -> str:
    image = np.full((4, 6, 3), value, dtype=np.uint8)
    write_image_png(path, image)
    return sha256_file(path)


# --- candidate files -------------------------------------------------------------------
def test_candidates_are_tracked_plus_untracked_files_that_are_not_ignored(
    gate_repo: Path,
) -> None:
    write_text(gate_repo / "notes 笔记.md", "new\n")
    write_text(gate_repo / ".local" / "ignored.md", "ignored\n")
    files = candidate_files(gate_repo)
    assert "notes 笔记.md" in files
    assert "README.md" in files
    assert not [f for f in files if f.startswith(".local/")]
    assert list(files) == sorted(files)


def test_ignored_files_are_never_scanned(gate_repo: Path, make_token: Callable[..., str]) -> None:
    key = make_token("s" + "k-", 48)
    write_text(gate_repo / ".local" / "scratch.md", f"key {key}\r\n")
    assert check_eol(gate_repo).passed
    assert check_credentials(gate_repo).passed


def test_root_must_be_the_top_of_a_git_work_tree(gate_repo: Path, unicode_tmp_path: Path) -> None:
    plain = unicode_tmp_path / "plain dir"
    plain.mkdir()
    for root in (gate_repo / "src", plain):
        with pytest.raises(GateError, match="work tree") as caught:
            candidate_files(root)
        assert caught.value.rule == "not-a-git-work-tree"


# --- eol -------------------------------------------------------------------------------
def test_eol_reports_the_first_crlf_line_and_the_count(gate_repo: Path) -> None:
    (gate_repo / "docs").mkdir()
    (gate_repo / "docs" / "a.txt").write_bytes(b"one\ntwo\r\nthree\r\n")
    assert lines(check_eol(gate_repo)) == ["eol docs/a.txt:2 crlf (2 CRLF line endings)"]


def test_eol_skips_binary_content_and_ignores_lone_cr(gate_repo: Path) -> None:
    (gate_repo / "blob.dat").write_bytes(b"\x00\x01\r\n\x02")
    (gate_repo / "old-mac.txt").write_bytes(b"a\rb\r")
    _png(gate_repo / "pic.png")
    result = check_eol(gate_repo)
    assert result.passed
    assert result.scanned == len(candidate_files(gate_repo)) - 2  # blob.dat, pic.png


def test_text_too_large_to_scan_fails(gate_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(repo_module, "TEXT_SCAN_LIMIT", 4096)
    write_text(gate_repo / "big.txt", "x" * 5000 + "\n")
    assert lines(check_credentials(gate_repo)) == [
        "credentials big.txt too-large-to-scan (5001 bytes)"
    ]


# --- credentials (file names) ----------------------------------------------------------
@pytest.mark.parametrize(
    ("rel", "rule"),
    [
        ("kaggle.json", "kaggle-json-file"),
        ("config/.env", "dotenv-file"),
        (".env.production", "dotenv-file"),
        ("keys/id_ed25519", "private-key-file"),
        ("certs/server.pem", "private-key-file"),
        (".netrc", "credentials-file"),
    ],
)
def test_credential_store_file_names_fail(gate_repo: Path, rel: str, rule: str) -> None:
    write_text(gate_repo / rel, "nothing secret in here\n")
    assert lines(check_credentials(gate_repo)) == [f"credentials {rel} {rule}"]


def test_env_templates_are_allowed(gate_repo: Path) -> None:
    write_text(gate_repo / ".env.example", "OPENAI_API_KEY=<your key>\n")
    assert check_credentials(gate_repo).passed


# --- media -----------------------------------------------------------------------------
def _media_toml(entries: list[dict[str, object]]) -> str:
    out = ["schema_version = 1", ""]
    for entry in entries:
        out.append("[[media]]")
        for key, value in entry.items():
            out.append(f"{key} = {json.dumps(value)}")
        out.append("")
    return "\n".join(out)


def test_registered_media_passes_and_changed_bytes_fail(gate_repo: Path) -> None:
    rel = "tests/fixtures/golden/frame.png"
    digest = _png(gate_repo / rel)
    write_text(
        gate_repo / "registries" / "media.toml",
        _media_toml([{"path": rel, "sha256": digest, "restrictions": []}]),
    )
    assert check_media(gate_repo).passed
    _png(gate_repo / rel, value=7)
    assert lines(check_media(gate_repo)) == [f"media {rel} sha256-mismatch"]


def test_media_with_blocked_restrictions_fails_even_with_a_matching_hash(gate_repo: Path) -> None:
    digest = _png(gate_repo / "a.png")
    tags = ["no_share", "derived_aggregate", "nda"]
    write_text(
        gate_repo / "registries" / "media.toml",
        _media_toml([{"path": "a.png", "sha256": digest, "restrictions": tags}]),
    )
    assert lines(check_media(gate_repo)) == [
        "media a.png restricted-tag (nda)",
        "media a.png restricted-tag (no_share)",
    ]


def test_media_suffixes_cover_data_weights_archives_and_video(gate_repo: Path) -> None:
    rels = ["a/x.npy", "a/y.parquet", "a/w.safetensors", "a/z.zip", "a/v.mp4", "a/s.wav"]
    for rel in rels:
        (gate_repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (gate_repo / rel).write_bytes(b"\x00payload")
    result = check_media(gate_repo)
    assert lines(result) == [f"media {rel} unregistered" for rel in sorted(rels)]


def test_media_registry_problems_are_reported_per_file(gate_repo: Path) -> None:
    a = _png(gate_repo / "a.png")
    _png(gate_repo / "b.png", value=3)
    write_text(
        gate_repo / "registries" / "media.toml",
        _media_toml([{"path": "a.png", "sha256": a}, {"path": "./a.png", "sha256": a}])
        + '\n[[media]]\npath = "b.png"\n',
    )
    assert lines(check_media(gate_repo)) == [
        "media a.png registered-twice (2 entries)",
        "media b.png sha256-missing",
    ]
    write_text(gate_repo / "registries" / "media.toml", "media = [\n")
    assert lines(check_media(gate_repo)) == [
        "media a.png media-registry-unreadable",
        "media b.png media-registry-unreadable",
    ]


# --- restricted data -------------------------------------------------------------------
@pytest.mark.parametrize(
    ("rel", "rules"),
    [
        ("open/events/3788741.json", ["statsbomb-path"]),
        ("open/lineups/3788741.json", ["statsbomb-path"]),
        ("open/three-sixty/3788741.json", ["statsbomb-path"]),
        ("open/matches/11/90.json", ["statsbomb-path"]),
        ("open/matches/43.json", ["statsbomb-path"]),
        ("sn/england_epl/Labels-v2.json", ["soccernet-labels"]),
        ("sn/Labels-GameState.json", ["soccernet-labels"]),
        ("runs/align-1/align.json", ["under-runs"]),
        ("src/demo/data/table.csv", ["under-data"]),
        ("data/events/7.json", ["statsbomb-path", "under-data"]),
        ("cache/state.pkl", ["pickle-file"]),
        ("models/state.pkl.gz", ["pickle-file"]),
        ("out/evidence.jsonl", ["evidence-store"]),
        ("out/evidence/cards.parquet", ["evidence-store"]),
        ("cases/123.json", ["nda-case-file"]),
        ("private/cases/2026/7.json", ["nda-case-file"]),
        ("identity_overrides.json", ["identity-overrides"]),
        ("config/identity-overrides.v2.jsonl", ["identity-overrides"]),
        ("labels/manual_highlights_v1.json", ["manual-labels", "under-labels"]),
        ("eval/Labels/notes.txt", ["under-labels"]),
        ("notes/manual-moments.json", ["manual-labels"]),
        ("plugin/detections.jsonl", ["raw-plugin-output"]),
        ("sngs/calib.jsonl", ["raw-plugin-output"]),
        ("plugin/ball_raw.jsonl", ["raw-plugin-output"]),
        ("plugin/keypoints.jsonl", ["raw-plugin-output"]),
        ("plugin/keypoints-receipt.json", ["raw-plugin-output"]),
        ("job/Plugin-Receipt.json", ["raw-plugin-output"]),
        # Review findings: the ball receipt, renamed raw outputs, a plugin job directory,
        # other label directories, identity-override variants and every file below cases.
        ("plugin/ball-receipt.json", ["raw-plugin-output"]),
        ("plugin/detections.json", ["raw-plugin-output"]),
        ("plugin/detections_v2.jsonl", ["raw-plugin-output"]),
        ("plugin/sngs_detections.jsonl", ["raw-plugin-output"]),
        ("plugin/ball-raw.jsonl", ["raw-plugin-output"]),
        ("plugin/ball_raw.ndjson", ["raw-plugin-output"]),
        ("calibration-0123456789abcdef/out/tracks.jsonl", ["under-plugin-job"]),
        ("eval/manual_labels/highlights.json", ["under-labels"]),
        ("eval/label/highlights.json", ["under-labels"]),
        ("cfg/identity_override.json", ["identity-overrides"]),
        ("cfg/identity_overrides.yaml", ["identity-overrides"]),
        ("cases/42.jsonl", ["nda-case-file"]),
        # The same rules as patterns, beyond the names the review tried.
        ("plugin/plugin-receipt.v2.json", ["raw-plugin-output"]),
        ("plugin/calib_v2.ndjson", ["raw-plugin-output"]),
        ("plugin/detections.jsonl.txt", ["raw-plugin-output"]),
        ("calibration-0123456789abcdef/job.json", ["under-plugin-job"]),
        ("eval/Manual-Labels_v2/moments.csv", ["under-labels"]),
        ("cfg/match7_identity-overrides", ["identity-overrides"]),
        ("private/cases/notes.md", ["nda-case-file"]),
        # Second review: copies, named the way browsers, file managers and people name
        # them, passed although .gitignore hid them, and so did data files named like JSON
        # Schemas. The last ones follow the patterns of .gitignore further.
        ("plugin/detections (1).jsonl", ["raw-plugin-output"]),
        ("plugin/detections - Copy.jsonl", ["raw-plugin-output"]),
        ("plugin/detections copy.jsonl", ["raw-plugin-output"]),
        ("plugin/detections2.jsonl", ["raw-plugin-output"]),
        ("plugin/detectionsV2.json", ["raw-plugin-output"]),
        ("plugin/Copy of detections.jsonl", ["raw-plugin-output"]),
        ("plugin/ball_raw (1).jsonl", ["raw-plugin-output"]),
        ("plugin/ball_raw2.jsonl", ["raw-plugin-output"]),
        ("plugin/ball-receipt (1).json", ["raw-plugin-output"]),
        ("plugin/calib (1).jsonl", ["raw-plugin-output"]),
        ("plugin/keypoints.ldjson", ["raw-plugin-output"]),
        ("cfg/identity_overrides (1).json", ["identity-overrides"]),
        ("cfg/identity_overrides2.json", ["identity-overrides"]),
        ("cfg/identity_overrides - Copy.json", ["identity-overrides"]),
        ("cfg/identity_overrides.schema.json", ["identity-overrides"]),
        ("plugin/detections.schema.json", ["raw-plugin-output"]),
        ("notes/manual_moments.ldjson", ["manual-labels"]),
        ("_scratch-0123456789abcdef/out/tracks.jsonl", ["under-plugin-job"]),
    ],
)
def test_restricted_data_rules(gate_repo: Path, rel: str, rules: list[str]) -> None:
    path = gate_repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"{}\n")
    result = check_restricted_data(gate_repo)
    assert sorted(p.rule for p in result.problems if p.path == rel) == rules
    assert all(p.path == rel for p in result.problems)


def test_statsbomb_content_is_sniffed_in_the_first_64_kib_only(gate_repo: Path) -> None:
    event = {"possession_team": {"id": 1}, "play_pattern": {"id": 1}}
    write_text(gate_repo / "fixtures" / "sample.json", json.dumps([event]))
    late = {"padding": "x" * 70_000, **event}
    write_text(gate_repo / "fixtures" / "late.json", json.dumps(late))
    assert lines(check_restricted_data(gate_repo)) == [
        "restricted_data fixtures/sample.json statsbomb-content"
    ]


JSON_SCHEMA = {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"}


def test_restricted_data_leaves_code_docs_and_registries_alone(gate_repo: Path) -> None:
    for rel in (
        "src/demo/evidence.py",
        "docs/data-access.md",
        "registries/data.toml",
        "tests/fixtures/events.json",
        "open/matches/latest.json",
        "docs/runs.md",
        "src/demo/label_eval.py",
        "src/demo/identity_overrides.py",
        "tests/test_labels.py",
        "docs/cases.md",
        "docs/manual_review.md",
        "docs/detections.jsonl.md",
        # Code and docs named like restricted material, and directories whose names only
        # start like a restricted one.
        "docs/identity-overrides.md",
        "src/demo/ball_raw.py",
        "src/demo/detections/__init__.py",
        "src/demo/manual_labeling/__init__.py",
        "src/demo/labeling/cases_report.py",
        "docs/plugin-jobs/calibration-job.md",
        # The project's schema directory: its .schema.json files are JSON Schemas.
        "src/gbf/contracts/schemas/detections.schema.json",
    ):
        write_text(gate_repo / rel, "# placeholder\n")
    # This test used to write placeholders into the schemas below too. Since a data file
    # named like a schema passed (second review), the name rules skip a .schema.json file
    # outside the schema directory only when it is a JSON Schema, with a "$schema" key.
    for rel in (
        "src/demo/schemas/plugin-receipt.schema.json",
        "src/demo/schemas/identity_overrides.schema.json",
        "src/demo/schemas/detections.schema.json",
    ):
        write_json(gate_repo / rel, JSON_SCHEMA)
    assert check_restricted_data(gate_repo).passed


def test_restricted_tags_count_in_json_restrictions_arrays_near_the_top(gate_repo: Path) -> None:
    out = gate_repo / "out"
    write_json(out / "tagged.json", {"restrictions": ["no_share", "hl3", "nda"], "value": 1})
    (out / "bom.geojson").write_bytes(codecs.BOM_UTF8 + b'{"restrictions": ["research_nc"]}\n')
    write_json(out / "string.json", {"restrictions": "no_identity"})
    # A JSON Schema lists every tag in an enum; deeper and non-blocked tags pass too.
    schema = {
        "$id": "urn:example",
        "properties": {"restrictions": {"items": {"enum": ["nda", "no_share", "research_nc"]}}},
    }
    write_json(out / "schema.json", schema)
    write_json(out / "nested.json", {"items": [{"restrictions": ["no_share"]}]})
    write_json(out / "deep.json", {"run": {"stage": {"restrictions": ["nda"]}}})
    write_json(out / "public.json", {"restrictions": ["derived_aggregate", "non_commercial"]})
    # This test used to assert that a top-level array passes. Such an array is what
    # json.dump(rows) writes for a list of tagged records, so passing it let a whole tagged
    # export through (review finding); its first elements are now read like objects.
    write_json(out / "list.json", [{"restrictions": ["nda"]}])
    count = 32  # JSON_ARRAY_SNIFF_ELEMENTS: later elements are not read
    late = [{"frame": i, "restrictions": []} for i in range(count)] + [{"restrictions": ["nda"]}]
    write_json(out / "late.json", late)
    # This test also used to assert that unparseable JSON passes, even when it mentions
    # restrictions; such a file cannot be scanned, so it now fails closed.
    write_text(out / "broken.json", '{"restrictions": ["nda"\n')
    assert lines(check_restricted_data(gate_repo)) == [
        "restricted_data out/bom.geojson restricted-tag (research_nc)",
        "restricted_data out/broken.json:2 json-parse-error",
        "restricted_data out/list.json restricted-tag (nda)",
        "restricted_data out/string.json restricted-tag (no_identity)",
        "restricted_data out/tagged.json restricted-tag (nda)",
        "restricted_data out/tagged.json restricted-tag (no_share)",
    ]


@pytest.mark.parametrize(
    "document",
    [
        pytest.param({"run": {"id": "r1", "restrictions": ["no_share"]}}, id="one-level-down"),
        pytest.param(
            [{"frame": 0, "restrictions": []}, {"frame": 1, "restrictions": ["no_share"]}],
            id="top-level-array",
        ),
        pytest.param({"Restrictions": ["No_Share"]}, id="key-in-another-case"),
        # Second review: a string that holds several tags passed.
        pytest.param({"restrictions": "hl3, no_share"}, id="string-of-tags"),
        pytest.param({"run": {"restrictions": "hl3 no_share"}}, id="string-one-level-down"),
        pytest.param([{"restrictions": ["hl3; no_share"]}], id="list-item-of-tags"),
    ],
)
def test_restricted_tags_where_exports_put_them(gate_repo: Path, document: object) -> None:
    write_json(gate_repo / "out" / "export.json", document)
    write_text(gate_repo / "out" / "export.jsonl", json.dumps(document) + "\n")
    assert lines(check_restricted_data(gate_repo)) == [
        "restricted_data out/export.json restricted-tag (no_share)",
        "restricted_data out/export.jsonl restricted-tag (no_share)",
    ]


def test_json_lines_whose_first_records_overflow_the_sniff_window_fail_closed(
    gate_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(datafiles_module, "JSON_LINES_SNIFF_BYTES", 4096)
    big = json.dumps({"restrictions": ["no_share"], "pad": "x" * 5000}) + "\n"
    write_text(gate_repo / "out" / "big-first.jsonl", big * 3)
    write_text(gate_repo / "out" / "big-second.jsonl", json.dumps({"frame": 0}) + "\n" + big)
    # Short records overflow the window too, but its first 32 records are read whole.
    rows = "".join(json.dumps({"frame": i, "restrictions": []}) + "\n" for i in range(200))
    write_text(gate_repo / "out" / "long.jsonl", rows)
    assert lines(check_restricted_data(gate_repo)) == [
        "restricted_data out/big-first.jsonl too-large-to-scan "
        "(0 of 32 records in the first 4096 bytes)",
        "restricted_data out/big-second.jsonl too-large-to-scan "
        "(1 of 32 records in the first 4096 bytes)",
    ]


def test_json_that_cannot_be_decoded_or_parsed_fails_closed(gate_repo: Path) -> None:
    out = gate_repo / "out"
    out.mkdir()
    (out / "latin1.json").write_bytes(b'{"restrictions": ["no_share"], "league": "S\xe9rie A"}\n')
    write_text(out / "trailing-comma.json", '{"restrictions": ["no_share"], "a": 1,}\n')
    rows = [json.dumps({"frame": 0}), '{"frame": 1, "restrictions": ["no_share"]', "{}"]
    write_text(out / "broken.jsonl", "\n".join(rows) + "\n")
    (out / "latin1.jsonl").write_bytes(b'{"frame": 0}\n{"league": "S\xe9rie A"}\n')
    # Text that does not parse and never mentions restrictions carries no tags to find.
    write_text(out / "settings.json", '{\n  // editor settings\n  "tabSize": 4,\n}\n')
    assert lines(check_restricted_data(gate_repo)) == [
        "restricted_data out/broken.jsonl:2 json-parse-error",
        "restricted_data out/latin1.json undecodable",
        "restricted_data out/latin1.jsonl:2 undecodable",
        "restricted_data out/trailing-comma.json:1 json-parse-error",
    ]


def test_restricted_tags_in_the_first_records_of_json_lines(gate_repo: Path) -> None:
    clean = [{"frame": i, "restrictions": ["agpl_network_source"]} for i in range(3)]
    tagged = [*clean, {"frame": 3, "restrictions": ["agpl_network_source", "no_identity"]}]
    write_text(gate_repo / "out" / "rows.jsonl", "".join(json.dumps(r) + "\n" for r in tagged))
    write_text(gate_repo / "out" / "rows.ndjson", "\n" + json.dumps({"restrictions": ["nda"]}))
    write_text(gate_repo / "out" / "clean.jsonl", "".join(json.dumps(r) + "\n" for r in clean))
    count = datafiles_module.JSON_LINES_SNIFF_RECORDS
    late = [{"frame": i, "restrictions": []} for i in range(count)] + [{"restrictions": ["nda"]}]
    write_text(gate_repo / "out" / "late.jsonl", "".join(json.dumps(r) + "\n" for r in late))
    assert lines(check_restricted_data(gate_repo)) == [
        "restricted_data out/rows.jsonl restricted-tag (no_identity)",
        "restricted_data out/rows.ndjson restricted-tag (nda)",
    ]


#: Hand labels in the shape gbf.label_eval reads (synthetic).
HAND_LABELS = {
    "id": "synthetic-labels",
    "moments": [{"id": "M1", "type": "duel_tackle", "start": 100, "peak": 110, "end": 120}],
    "cuts": [50],
}


def test_hand_labels_fail_under_any_name(gate_repo: Path) -> None:
    # Fourth review: hand labels carry no restrictions key, so only their name protected
    # them, and a copy under another name passed. They are recognised by their content
    # now, where the restricted-tag rule reads a JSON document.
    notes = gate_repo / "notes"
    write_json(notes / "highlights_v1.json", HAND_LABELS)
    write_text(notes / "copy.jsonc", json.dumps(HAND_LABELS, indent=2) + "\n")
    write_text(notes / "rows.jsonl", json.dumps({"labels": HAND_LABELS}) + "\n")
    write_json(notes / "batch.json", [{"clip": "a"}, HAND_LABELS])
    write_json(notes / "manual_highlights_v2.json", HAND_LABELS)
    # What gbf writes about its highlights, and moments without a key the labels need, are
    # not hand labels.
    highlight = {"id": "H1", "type": "duel_tackle", "start_frame": 100, "peak_frame": 110}
    write_json(notes / "final.json", {"highlights": [dict(highlight, end_frame=120)]})
    write_json(notes / "run.json", {"result": {"moments": [dict(highlight, tier="B")]}})
    partial = {k: v for k, v in HAND_LABELS["moments"][0].items() if k != "id"}
    write_json(notes / "partial.json", {"moments": [partial]})
    assert lines(check_restricted_data(gate_repo)) == [
        "restricted_data notes/batch.json manual-labels-content",
        "restricted_data notes/copy.jsonc manual-labels-content",
        "restricted_data notes/highlights_v1.json manual-labels-content",
        "restricted_data notes/manual_highlights_v2.json manual-labels",
        "restricted_data notes/manual_highlights_v2.json manual-labels-content",
        "restricted_data notes/rows.jsonl manual-labels-content",
    ]


def test_hand_labels_are_recognised_by_the_keys_label_eval_needs() -> None:
    # Labels without one of these keys do not load, so the gate misses no labels that do.
    no_highlights = {"highlights": []}
    assert label_eval.evaluate(HAND_LABELS, no_highlights)["n_labels"] == 1
    for key in sorted(datafiles_module.HAND_LABEL_MOMENT_KEYS):
        moment = {k: v for k, v in HAND_LABELS["moments"][0].items() if k != key}
        with pytest.raises(label_eval.LabelEvalError, match=f"missing '{key}'"):
            label_eval.evaluate({"moments": [moment]}, no_highlights)


def test_html_with_the_private_banner_or_a_restrictions_statement_fails(gate_repo: Path) -> None:
    docs = gate_repo / "docs"
    write_text(docs / "banner.html", "<!doctype html>\n<p><strong>私有素材 · 禁止分享</strong>\n")
    entity_banner = "".join(f"&#{ord(c)};" for c in "私有素材")
    write_text(docs / "entities.htm", f"<p>{entity_banner}</p>\n")
    write_text(
        docs / "embedded.html",
        "<pre>{&quot;restrictions&quot;: [&quot;hl3&quot;, &quot;research_nc&quot;]}</pre>\n",
    )
    write_text(docs / "attribute.html", '<main data-restrictions="no_share">x</main>\n')
    write_text(docs / "statement.html", "<li>限制标签：agpl_network_source、hl3、nda</li>\n")
    write_text(
        docs / "public.html",
        "<p>Every artifact carries restrictions; no_share marks private material.</p>\n"
        '<pre>{"restrictions": ["derived_aggregate"]}</pre>\n<p>限制标签：无</p>\n',
    )
    write_text(docs / "notes.md", '私有素材 and `"restrictions": ["nda"]` in prose\n')
    assert lines(check_restricted_data(gate_repo)) == [
        "restricted_data docs/attribute.html restricted-tag (no_share)",
        "restricted_data docs/banner.html private-banner",
        "restricted_data docs/embedded.html restricted-tag (research_nc)",
        "restricted_data docs/entities.htm private-banner",
        "restricted_data docs/statement.html restricted-tag (nda)",
    ]


@pytest.mark.parametrize(
    ("name", "page", "problems"),
    [
        pytest.param(
            "table.html",
            "<table><tr><th>restrictions</th><td>no_share</td></tr></table>\n",
            ["restricted-tag (no_share)"],
            id="table-cell",
        ),
        pytest.param(
            "meta.html",
            '<meta name="restrictions" content="no_share">\n',
            ["restricted-tag (no_share)"],
            id="meta-tag",
        ),
        pytest.param(
            "english.html",
            "<p>Restrictions: no_share, nda</p>\n",
            ["restricted-tag (nda)", "restricted-tag (no_share)"],
            id="english-label",
        ),
        pytest.param(
            "split.html",
            "<p>私有<b>素材</b></p>\n",
            ["private-banner"],
            id="banner-split-by-markup",
        ),
        pytest.param("report.xhtml", "<p>私有素材</p>\n", ["private-banner"], id="xhtml"),
        pytest.param(
            "card.svg",
            "<svg><text>私有素材 · 禁止分享</text></svg>\n",
            ["private-banner"],
            id="svg",
        ),
        # Further shapes of the same statements.
        pytest.param(
            "columns.html",
            "<table><tr><th>clip</th><th>Restrictions</th></tr>\n"
            "<tr><td>a.mp4</td><td><code>research_nc</code></td></tr></table>\n",
            ["restricted-tag (research_nc)"],
            id="table-column",
        ),
        pytest.param(
            "terms.html",
            "<dl><dt>restrictions:</dt><dd>no_identity</dd></dl>\n",
            ["restricted-tag (no_identity)"],
            id="definition-list",
        ),
        pytest.param(  # "İ".lower() is two characters: offsets must not drift
            "istanbul.html",
            "<p>" + "İ" * 64 + "</p><table><tr><th>restrictions</th><td>nda</td></tr></table>\n",
            ["restricted-tag (nda)"],
            id="table-after-text-whose-lower-case-is-longer",
        ),
        pytest.param(
            "bold.html",
            "<p>\n  <b>Restrictions:</b>\n  hl3, no_share\n</p>\n",
            ["restricted-tag (no_share)"],
            id="label-in-markup",
        ),
        pytest.param(
            "zh.html",
            "<li>限制标签：<code>nda</code></li>\n",
            ["restricted-tag (nda)"],
            id="zh-label-in-markup",
        ),
        # Second review: shapes that passed.
        pytest.param(
            "zh-table.html",
            "<table><tr><th>限制标签</th><td>no_share</td></tr></table>\n",
            ["restricted-tag (no_share)"],
            id="zh-table-cell",
        ),
        pytest.param(
            "zh-terms.html",
            "<dl><dt>限制标签</dt><dd>nda</dd></dl>\n",
            ["restricted-tag (nda)"],
            id="zh-definition-list",
        ),
        pytest.param(
            "tags-column.html",
            "<table><tr><th>Restriction tags</th><td>nda</td></tr></table>\n",
            ["restricted-tag (nda)"],
            id="restriction-tags-cell",
        ),
        pytest.param(
            "class.html",
            '<ul class="restrictions"><li>no_share</li><li>hl3</li></ul>\n',
            ["restricted-tag (no_share)"],
            id="list-with-a-restrictions-class",
        ),
        pytest.param(
            "itemprop.html",
            '<p>Tags: <span itemprop="restrictions">research_nc</span></p>\n',
            ["restricted-tag (research_nc)"],
            id="element-named-restrictions",
        ),
        pytest.param(
            "list-below.html",
            "<p>Restrictions:</p>\n<ul>\n  <li>hl3</li>\n  <li>no_share</li>\n</ul>\n",
            ["restricted-tag (no_share)"],
            id="list-below-a-label",
        ),
        pytest.param(
            "zh-list-below.html",
            "<p>限制标签：</p><ul><li>nda</li></ul>\n",
            ["restricted-tag (nda)"],
            id="list-below-a-zh-label",
        ),
        pytest.param(
            "semicolon.html",
            "<p>Restrictions: hl3; no_share</p>\n",
            ["restricted-tag (no_share)"],
            id="semicolon-separated",
        ),
        pytest.param(
            "restriction-tags.html",
            "<p>Restriction tags: no_identity</p>\n",
            ["restricted-tag (no_identity)"],
            id="restriction-tags-label",
        ),
        pytest.param(
            "labels.svg",
            "<svg><text>Restrictions</text><text>no_share</text><text>clip 7</text></svg>\n",
            ["restricted-tag (no_share)"],
            id="svg-text-after-a-label",
        ),
        pytest.param(
            "zwsp.html",
            "<p>私\N{ZERO WIDTH SPACE}有素材</p>\n",
            ["private-banner"],
            id="banner-split-by-a-zero-width-space",
        ),
        pytest.param(
            "shy.html", "<p>私&shy;有素材</p>\n", ["private-banner"], id="banner-split-by-shy"
        ),
        pytest.param(
            "wbr.html",
            "<p>Restrictions: research_<wbr>nc</p>\n",
            ["restricted-tag (research_nc)"],
            id="tag-split-by-wbr",
        ),
        pytest.param(
            "zwsp-tag.html",
            "<p>Restrictions: no_\N{ZERO WIDTH SPACE}share</p>\n",
            ["restricted-tag (no_share)"],
            id="tag-split-by-a-zero-width-space",
        ),
        # Third review: a string after a quoted key, and label words split by <wbr>.
        pytest.param(
            "json-string.html",
            '<pre>{"restrictions": "no_share"}</pre>\n',
            ["restricted-tag (no_share)"],
            id="json-string-in-a-page",
        ),
        pytest.param(
            "escaped-json-string.html",
            "<pre>{&quot;restrictions&quot;: &quot;nda&quot;}</pre>\n",
            ["restricted-tag (nda)"],
            id="escaped-json-string",
        ),
        pytest.param(
            "json-ld.html",
            '<script type="application/ld+json">{"restrictions": "hl3, research_nc"}</script>\n',
            ["restricted-tag (research_nc)"],
            id="json-ld-string",
        ),
        pytest.param(
            "wbr-label.html",
            "<p>Re<wbr>strictions: nda</p>\n",
            ["restricted-tag (nda)"],
            id="label-split-by-wbr",
        ),
        pytest.param(
            "wbr-zh-label.html",
            "<p>限制<wbr>标签：no_share</p>\n",
            ["restricted-tag (no_share)"],
            id="zh-label-split-by-wbr",
        ),
        pytest.param(
            "wbr-table.html",
            "<table><tr><th>Re<wbr>strictions</th><td>no_identity</td></tr></table>\n",
            ["restricted-tag (no_identity)"],
            id="table-cell-split-by-wbr",
        ),
        pytest.param(
            "wbr-labels.svg",
            "<svg><text>Restric<wbr/>tions</text><text>nda</text></svg>\n",
            ["restricted-tag (nda)"],
            id="svg-label-split-by-wbr",
        ),
        # Fourth review: the string after a quoted key or attribute stopped at the first
        # quote of either kind, also at an escaped one, so the tags after it passed.
        pytest.param(
            "apostrophe.html",
            '<pre>{"restrictions": "partner\'s nda"}</pre>\n',
            ["restricted-tag (nda)"],
            id="apostrophe-in-a-json-string",
        ),
        pytest.param(
            "escaped-quotes.html",
            '<pre>{"restrictions": "hl3 \\"x\\" no_share"}</pre>\n',
            ["restricted-tag (no_share)"],
            id="escaped-quotes-in-a-json-string",
        ),
        pytest.param(
            "single-quoted.html",
            "<pre>{'restrictions': 'say \"no\", research_nc'}</pre>\n",
            ["restricted-tag (research_nc)"],
            id="double-quotes-in-a-single-quoted-string",
        ),
        pytest.param(
            "attribute-apostrophe.html",
            '<main data-restrictions="partner\'s no_identity">x</main>\n',
            ["restricted-tag (no_identity)"],
            id="apostrophe-in-an-attribute",
        ),
    ],
)
def test_html_restriction_statements_in_other_shapes_fail(
    gate_repo: Path, name: str, page: str, problems: list[str]
) -> None:
    write_text(gate_repo / "docs" / name, page)
    expected = [f"restricted_data docs/{name} {problem}" for problem in problems]
    assert lines(check_restricted_data(gate_repo)) == expected


def test_html_that_only_talks_about_restrictions_passes(gate_repo: Path) -> None:
    write_text(
        gate_repo / "docs" / "guide.html",
        "<h2>Restrictions:</h2>\n<p>Every artifact carries tags such as no_share.</p>\n"
        "<p>Restrictions: none. Tags like nda are explained in ADR 0002.</p>\n"
        "<table><tr><th>field</th><td>restrictions, a list of tags (no_share, nda)</td></tr>"
        "</table>\n"
        '<meta name="description" content="What no_share and nda mean">\n',
    )
    write_text(
        gate_repo / "docs" / "flow.svg",
        "<svg><text>restrictions flow from inputs (no_share) to outputs</text></svg>\n",
    )
    assert check_restricted_data(gate_repo).passed


def test_text_near_a_label_that_holds_more_than_tags_passes(gate_repo: Path) -> None:
    # The shapes the second review added count only when the text holds nothing but tags.
    docs = gate_repo / "docs"
    write_text(
        docs / "glossary.html",
        "<h2>Restrictions:</h2>\n<ul><li>no_share: never leaves this machine</li>"
        "<li>nda: a case file under an NDA</li></ul>\n"
        "<h2>Restriction tags</h2>\n<ul><li>nda</li><li>no_share</li></ul>\n"
        "<p>The gate blocks these restrictions:</p>\n<ul><li>nda</li></ul>\n"
        '<section id="restrictions"><h3>Restrictions</h3>'
        "<p>Tags such as no_share and nda travel with every artifact.</p></section>\n"
        "<p>Restrictions: none; tags like nda are explained in ADR 0002.</p>\n"
        '<meta name="restrictions" content="hl3"><title>What nda means</title>\n',
    )
    write_text(
        docs / "legend.svg",
        "<svg><text>Restrictions</text><text>flow from inputs (no_share)</text>"
        "<text>nda</text></svg>\n",
    )
    write_text(
        docs / "report.html",
        "<ul><li>输入没有标注限制标签；"
        "发布前仍须按 ADR 0002 核对画面和数据的来源权利。</li></ul>\n",
    )
    # A string after a quoted key ends at its closing quote, as in JSON (third review), and
    # an escaped quote does not end it (fourth review).
    write_text(
        docs / "example.html",
        '<pre>{"restrictions": "hl3", "note": "how nda material is handled"}</pre>\n'
        '<pre>{"restrictions": "hl3 \\"x\\"", "note": "how nda material is handled"}</pre>\n',
    )
    write_text(
        gate_repo / "out" / "notes.jsonc",
        '{\n  // exported by hand\n  "restrictions": "hl3", "note": "no nda material"\n}\n',
    )
    assert check_restricted_data(gate_repo).passed


def test_restricted_content_in_other_file_kinds_fails(gate_repo: Path) -> None:
    # Second review: these kinds of file were not read at all.
    out = gate_repo / "out"
    write_text(out / "hand.json5", "{\n  // exported by hand\n  restrictions: ['no_share'],\n}\n")
    write_text(out / "tagged.jsonc", '{"restrictions": ["nda"]}\n')
    write_text(out / "rows.ldjson", json.dumps({"frame": 0, "restrictions": ["research_nc"]}))
    write_text(out / "report.xml", "<report><restrictions>no_identity</restrictions></report>\n")
    write_text(out / "report.html.bak", "<p>Restrictions: nda</p>\n")
    write_text(out / "card.svg~", "<svg><text>私有素材</text></svg>\n")
    # Saved pages: a declared encoding, base64 in a multipart message, and quoted-printable
    # that no header declares.
    banner = quopri.encodestring("<p>私有素材</p>".encode()).decode("ascii")
    write_text(
        out / "declared.mhtml",
        "MIME-Version: 1.0\nContent-Type: text/html; charset=utf-8\n"
        f"Content-Transfer-Encoding: quoted-printable\n\n{banner}\n",
    )
    statement = base64.b64encode("<p>限制标签：hl3、no_share</p>".encode()).decode("ascii")
    write_text(
        out / "multipart.mht",
        'MIME-Version: 1.0\nContent-Type: multipart/related; boundary="b"\n\n--b\n'
        "Content-Type: text/html; charset=utf-8\nContent-Transfer-Encoding: base64\n\n"
        f"{statement}\n--b--\n",
    )
    write_text(out / "bare.mhtml", f"MIME-Version: 1.0\n\n{banner}\n")
    write_text(out / "title.mht", "Subject: 私有素材\nMIME-Version: 1.0\n\n<p>ok</p>\n")
    # Third review: a string after a quoted key in a file with comments passed, and so did
    # one a level down.
    write_text(out / "string.jsonc", '{\n  // exported by hand\n  "restrictions": "nda"\n}\n')
    write_text(out / "run.jsonc", '{\n  // run 7\n  "run": {"restrictions": "no_share"},\n}\n')
    write_text(out / "quoted.json5", "{\n  'restrictions': 'research_nc',\n}\n")
    # A part declared as gb2312 is read as gb18030, as browsers read it: 珺 is only in GBK.
    gb2312 = "MIME-Version: 1.0\nContent-Type: text/html; charset=gb2312\n\n".encode("ascii")
    (out / "gbk-banner.mhtml").write_bytes(gb2312 + "<p>私有素材 珺</p>\n".encode("gbk"))
    (out / "gbk-text.mhtml").write_bytes(gb2312 + "<p>赛事 珺</p>\n".encode("gbk"))
    # Comments are legal in .json5 and .jsonc: a file that only mentions restrictions in
    # one passes, and so does a saved page without statements.
    write_text(out / "settings.jsonc", '{\n  // restrictions: see ADR 0002\n  "tabSize": 4,\n}\n')
    write_text(out / "public.mhtml", "MIME-Version: 1.0\nContent-Type: text/html\n\n<p>ok</p>\n")
    assert lines(check_restricted_data(gate_repo)) == [
        "restricted_data out/bare.mhtml private-banner",
        "restricted_data out/card.svg~ private-banner",
        "restricted_data out/declared.mhtml private-banner",
        "restricted_data out/gbk-banner.mhtml private-banner",
        "restricted_data out/hand.json5 restricted-tag (no_share)",
        "restricted_data out/multipart.mht restricted-tag (no_share)",
        "restricted_data out/quoted.json5 restricted-tag (research_nc)",
        "restricted_data out/report.html.bak restricted-tag (nda)",
        "restricted_data out/report.xml restricted-tag (no_identity)",
        "restricted_data out/rows.ldjson restricted-tag (research_nc)",
        "restricted_data out/run.jsonc restricted-tag (no_share)",
        "restricted_data out/string.jsonc restricted-tag (nda)",
        "restricted_data out/tagged.jsonc restricted-tag (nda)",
        "restricted_data out/title.mht private-banner",
    ]


def test_a_restrictions_string_runs_to_the_quote_that_closes_it(gate_repo: Path) -> None:
    # Fourth review: in a file with comments, read as text, the string after a quoted key
    # stopped at the first quote of either kind, also at an escaped one.
    out = gate_repo / "out"
    write_text(
        out / "apostrophe.jsonc", '{\n  // by hand\n  "restrictions": "don\'t share: nda"\n}\n'
    )
    write_text(
        out / "escaped.jsonc", '{\n  // by hand\n  "restrictions": "hl3 \\"x\\" no_share"\n}\n'
    )
    write_text(out / "single.json5", "{\n  'restrictions': 'it\\'s research_nc',\n}\n")
    # What follows the closing quote is not part of the string.
    write_text(
        out / "public.jsonc",
        '{\n  // by hand\n  "restrictions": "hl3 \\"x\\"", "note": "no nda material"\n}\n',
    )
    assert lines(check_restricted_data(gate_repo)) == [
        "restricted_data out/apostrophe.jsonc restricted-tag (nda)",
        "restricted_data out/escaped.jsonc restricted-tag (no_share)",
        "restricted_data out/single.json5 restricted-tag (research_nc)",
    ]


def test_an_attribute_value_that_ends_in_a_backslash_closes_at_its_quote(
    gate_repo: Path,
) -> None:
    # Fifth review: read with escapes, the quote after the backslash did not close the
    # value, so these attributes passed; HTML has no escapes and they failed before the
    # fourth review's fix.
    docs = gate_repo / "docs"
    write_text(docs / "double.html", '<div data-restrictions="nda\\">x</div>\n')
    write_text(docs / "single.html", "<div data-restrictions='no_share\\'>x</div>\n")
    # What follows the closing quote is still not part of the value.
    write_text(docs / "after.html", '<div data-restrictions="hl3\\">no nda material</div>\n')
    assert lines(check_restricted_data(gate_repo)) == [
        "restricted_data docs/double.html restricted-tag (nda)",
        "restricted_data docs/single.html restricted-tag (no_share)",
    ]


@pytest.mark.parametrize(
    "header",
    [
        pytest.param('Content-Type: multipart/mixed; boundary="b{}"\n\n--b{}\n', id="multipart"),
        pytest.param("Content-Type: message/rfc822\n\n", id="message"),
    ],
)
def test_saved_pages_nested_too_deeply_to_parse_fail_closed(gate_repo: Path, header: str) -> None:
    # Third review: Python's email parser raised RecursionError on parts nested a thousand
    # levels deep, so the whole check failed as an internal error that named no file.
    nested = "".join(header.format(level, level) for level in range(1000))
    page = f"MIME-Version: 1.0\n{nested}Content-Type: text/html\n\n<p>ok</p>\n"
    write_text(gate_repo / "out" / "deep.mhtml", page)
    assert lines(check_restricted_data(gate_repo)) == [
        "restricted_data out/deep.mhtml mime-parse-error (parts nested too deeply)"
    ]


NUL_CHARSET = "ut" + chr(0) + "f-8"
STATEMENT = "<p>Restrictions: no_share</p>"


@pytest.mark.parametrize(
    ("parameter", "body"),
    [
        pytest.param(f'charset="{NUL_CHARSET}"', STATEMENT, id="nul-in-the-name"),
        pytest.param(  # an xn-- label that is not Punycode: UnicodeError, not a decode error
            'charset="idna"', STATEMENT + ".xn--", id="codec-that-fails-otherwise"
        ),
        # Fifth review: as an RFC 2231 parameter the NUL is in the charset that the
        # parameter's own value is written in, and get_content_charset, which decodes
        # that value, raised the ValueError outside the handler.
        pytest.param(f"charset*={NUL_CHARSET}''utf-8", STATEMENT, id="rfc-2231-nul"),
        pytest.param(f"charset*0*={NUL_CHARSET}''utf-8", STATEMENT, id="rfc-2231-continuation-nul"),
    ],
)
def test_saved_pages_with_an_unusable_charset_are_a_parse_problem(
    gate_repo: Path, parameter: str, body: str
) -> None:
    # Fourth review: decoding with such a charset raised ValueError, not
    # UnicodeDecodeError, so the whole check failed as an internal error that named no file
    # and hid the other findings. The part is still read as UTF-8: it is base64, so the
    # raw file does not show its statement (fifth review: skipping the part passed too).
    encoded = base64.b64encode(f"{body}\n".encode()).decode("ascii")
    write_text(
        gate_repo / "out" / "saved.mhtml",
        f"MIME-Version: 1.0\nContent-Type: text/html; {parameter}\n"
        f"Content-Transfer-Encoding: base64\n\n{encoded}\n",
    )
    write_json(gate_repo / "out" / "tagged.json", {"restrictions": ["nda"]})
    assert lines(check_restricted_data(gate_repo)) == [
        "restricted_data out/saved.mhtml mime-parse-error (unusable charset)",
        "restricted_data out/saved.mhtml restricted-tag (no_share)",
        "restricted_data out/tagged.json restricted-tag (nda)",
    ]


@pytest.mark.parametrize(
    "header",
    [
        pytest.param(f"multipart/mixed; boundary*={NUL_CHARSET}''b1", id="boundary-nul"),
        pytest.param(f"multipart/mixed; boundary*0*={NUL_CHARSET}''b1", id="boundary-nul-part"),
        pytest.param("multipart/mixed; boundary*=idna''b1", id="boundary-codec"),
        pytest.param(
            "multipart/mixed; boundary*=utf-8''b1; boundary*0*=utf-8''b1",
            id="boundary-conflicting-continuation",
        ),
    ],
)
@pytest.mark.parametrize("nested", [False, True])
def test_bad_mime_header_keeps_raw_and_other_file_findings(
    gate_repo: Path, header: str, nested: bool
) -> None:
    page = f"Content-Type: {header}\n\n--b1\n{STATEMENT}\n--b1--\n"
    if nested:
        page = (
            f'Content-Type: multipart/alternative; boundary="outer"\n\n--outer\n{page}\n--outer--\n'
        )
    write_text(gate_repo / "out" / "saved.mhtml", f"MIME-Version: 1.0\n{page}")
    write_json(gate_repo / "out" / "tagged.json", {"restrictions": ["nda"]})
    assert lines(check_restricted_data(gate_repo)) == [
        "restricted_data out/saved.mhtml mime-parse-error (unusable header parameter)",
        "restricted_data out/saved.mhtml restricted-tag (no_share)",
        "restricted_data out/tagged.json restricted-tag (nda)",
    ]


def test_conflicting_mime_charset_continuation_still_reads_decoded_body(gate_repo: Path) -> None:
    encoded = base64.b64encode(STATEMENT.encode()).decode("ascii")
    write_text(
        gate_repo / "out" / "saved.mhtml",
        "MIME-Version: 1.0\nContent-Type: text/html; "
        "charset*=utf-8''utf-8; charset*0*=utf-8''x\n"
        f"Content-Transfer-Encoding: base64\n\n{encoded}\n",
    )
    write_json(gate_repo / "out" / "tagged.json", {"restrictions": ["nda"]})
    assert lines(check_restricted_data(gate_repo)) == [
        "restricted_data out/saved.mhtml mime-parse-error (unusable charset)",
        "restricted_data out/saved.mhtml restricted-tag (no_share)",
        "restricted_data out/tagged.json restricted-tag (nda)",
    ]


def test_pathological_pages_scan_in_well_under_a_second(gate_repo: Path) -> None:
    # Second review: a long attribute name that no "=" follows was rescanned from each of
    # its characters, so the time grew with the square of its length (40 KB took 18 s;
    # this page took 39 s). It also holds unclosed labelled tables and label words
    # followed by runs of spaces.
    size = 64 * 1024
    long_name = "<a restriction" + "x" * size + ">\n"
    tables = "<table><tr><th>restrictions</th><td>x</td>" * (size // 42)
    spaces = ("restrictions" + " " * 50) * (size // 62)
    slow = long_name + tables + spaces
    # Fourth review: a fixed budget of 0.5 s failed once at 0.55 s on a loaded machine. The
    # page is now timed against an ordinary page of the same length with the same kinds of
    # content (attributes, labelled tables, labels), the two scanned alternately in this
    # process, best of three, so that load slows both alike. Unloaded, the slow page takes
    # about 0.04 s and the ordinary one 0.07 s; with the quadratic pattern put back, the slow
    # page took 73 s against 0.12 s. The ceiling of 2 s, about 50 times the unloaded time,
    # fails a slowdown of every page that large.
    row = '<tr><th class="restrictions">restrictions</th><td title="x">hl3</td></tr>\n'
    block = "<table>" + row * 8 + "</table>\n<p>Restrictions: hl3</p>\n"
    plain = (block * (len(slow) // len(block) + 1))[: len(slow)]
    write_text(gate_repo / "docs" / "slow.html", slow)
    write_text(gate_repo / "docs" / "plain.html", plain)
    times: dict[str, list[float]] = {"slow": [], "plain": []}
    for _ in range(3):
        for name, measured in times.items():
            workspace = Workspace(gate_repo, files=[f"docs/{name}.html"])
            start = time.perf_counter()
            result = check_restricted_data(gate_repo, workspace)
            measured.append(time.perf_counter() - start)
            assert result.passed, name
    slow_s, plain_s = min(times["slow"]), min(times["plain"])
    assert slow_s < max(5 * plain_s, 0.25), f"{slow_s:.2f} s, the ordinary page {plain_s:.2f} s"
    assert slow_s < 2.0, f"{slow_s:.2f} s"


def test_pages_without_a_label_word_or_wbr_skip_the_visible_text(
    gate_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Fourth review: the check for a label word built the visible text of every page, which
    # made pages without one 4 to 10 times slower to scan. Only <wbr> can join a label word
    # that the source does not hold, so the visible text is built for pages with one.
    built: list[int] = []
    visible_text = datafiles_module._visible_text

    def counting_visible_text(page: str) -> str:
        built.append(len(page))
        return visible_text(page)

    monkeypatch.setattr(datafiles_module, "_visible_text", counting_visible_text)
    write_text(gate_repo / "docs" / "plain.html", "<p>A <b>table</b> of clips.</p>\n" * 64)
    assert check_restricted_data(gate_repo).passed
    assert built == []
    write_text(gate_repo / "docs" / "wbr.html", "<p>Re<wbr>strictions: nda</p>\n")
    assert lines(check_restricted_data(gate_repo)) == [
        "restricted_data docs/wbr.html restricted-tag (nda)"
    ]
    assert built


DOTTED_I = "\N{LATIN CAPITAL LETTER I WITH DOT ABOVE}"
DOTLESS_I = "\N{LATIN SMALL LETTER DOTLESS I}"
LONG_S = "\N{LATIN SMALL LETTER LONG S}"


@pytest.mark.parametrize(
    "text",
    [
        "an ordinary page",
        "Restrictions: nda",
        "RESTRICTION TAGS",
        "限制标签：nda",
        f"RESTR{DOTTED_I}CT{DOTTED_I}ONS: nda",
        f"re{LONG_S}trictions",
        f"restr{DOTLESS_I}ctions",
        f"restr{DOTLESS_I}ct and {LONG_S}",
        "\N{KELVIN SIGN} restrict ion",
    ],
)
def test_the_quick_label_word_check_agrees_with_the_expression(text: str) -> None:
    # The speed-up of the fourth review searches substrings of the lower-case text, and
    # re.IGNORECASE also matches İ and ı to i and ſ to s, which str.lower() leaves alone.
    expected = datafiles_module._RESTRICTION_WORD.search(text) is not None
    assert datafiles_module._has_label_word(text, text.lower()) is expected


def test_only_the_folded_letters_spell_label_words_beyond_ascii() -> None:
    # The quick checks rely on this: re.IGNORECASE matches no character but _FOLDED_LETTERS
    # to a letter of "restriction" or "<wbr" besides its ASCII cases, and str.lower() turns
    # no character but İ into one (İ into "i" and a combining dot).
    letters = re.compile("[restricnowb]", re.IGNORECASE)
    folded, lowered = set(), set()
    for character in map(chr, range(128, sys.maxunicode + 1)):
        if letters.fullmatch(character):
            folded.add(character)
        if any(letter in "restricnowb</" for letter in character.lower()):
            lowered.add(character)
    assert folded == set(datafiles_module._FOLDED_LETTERS)
    assert lowered == {DOTTED_I}


def test_label_words_in_letters_that_ignorecase_folds_still_count(gate_repo: Path) -> None:
    write_text(gate_repo / "docs" / "dotted.html", f"<p>RESTR{DOTTED_I}CT{DOTTED_I}ONS: nda</p>\n")
    write_text(gate_repo / "docs" / "long-s.html", f"<p>Re{LONG_S}trictions: no_share</p>\n")
    assert lines(check_restricted_data(gate_repo)) == [
        "restricted_data docs/dotted.html restricted-tag (nda)",
        "restricted_data docs/long-s.html restricted-tag (no_share)",
    ]


def test_html_that_cannot_be_decoded_fails_closed(gate_repo: Path) -> None:
    (gate_repo / "docs").mkdir()
    (gate_repo / "docs" / "latin1.html").write_bytes(b"<p>S\xe9rie A</p>\n")
    (gate_repo / "docs" / "utf16.html").write_bytes("<p>私有素材</p>\n".encode("utf-16"))
    assert lines(check_restricted_data(gate_repo)) == [
        "restricted_data docs/latin1.html undecodable",
        "restricted_data docs/utf16.html private-banner",
    ]


def test_restricted_content_too_large_to_parse_fails(
    gate_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(datafiles_module, "TEXT_SCAN_LIMIT", 4096)
    record = json.dumps({"pad": "x" * 5000}) + "\n"
    page = "<p>" + "x" * 5000 + "</p>\n"
    write_text(gate_repo / "out" / "big.json", record)
    write_text(gate_repo / "out" / "big.html", page)
    write_text(gate_repo / "out" / "big.jsonl", record)  # only its first records are read
    assert lines(check_restricted_data(gate_repo)) == [
        f"restricted_data out/big.html too-large-to-scan ({len(page)} bytes)",
        f"restricted_data out/big.json too-large-to-scan ({len(record)} bytes)",
    ]


#: Restricted material as it is saved: the raw outputs, and copies and renames as
#: browsers, file managers and people make them. .gitignore hides each one and the gate
#: flags each one.
RESTRICTED_VARIANTS = [
    *(f"plugin/{name}" for name in sorted(datafiles_module.RAW_PLUGIN_OUTPUTS)),
    "plugin/detections.json",
    "plugin/detections_v2.jsonl",
    "plugin/sngs_detections.jsonl",
    "plugin/detections (1).jsonl",
    "plugin/detections - Copy.jsonl",
    "plugin/detections copy.jsonl",
    "plugin/detections2.jsonl",
    "plugin/detectionsV2.json",
    "plugin/Copy of detections.jsonl",
    "plugin/detections.csv",
    "plugin/ball-raw.jsonl",
    "plugin/ball_raw.ndjson",
    "plugin/ball_raw (1).jsonl",
    "plugin/ball_raw2.jsonl",
    "plugin/ball-receipt (1).json",
    "plugin/plugin-receipt.v2.json",
    "plugin/Copy of plugin-receipt.json",
    "plugin/keypoints_v2.jsonl",
    "plugin/keypoints.ldjson",
    "plugin/calib (1).jsonl",
    "plugin/calib-2.ndjson",
    "cfg/identity_overrides.json",
    "cfg/identity_override.json",
    "cfg/identity_overrides.yaml",
    "cfg/identity-overrides.toml",
    "cfg/identity_overrides (1).json",
    "cfg/identity_overrides2.json",
    "cfg/identity_overrides - Copy.json",
    "cfg/match7_identity_overrides.csv",
    "cfg/identity_overrides.schema.json",
    "plugin/detections.schema.json",
    "eval/manual_highlights_v1.json",
    "eval/manual-moments (1).jsonl",
    "cases/123.json",
    "cases/notes.md",
    "private/cases/2026/7.json",
    "labels/notes.txt",
    "eval/labels/manual_highlights_v1.json",
    "eval/label/highlights.json",
    "eval/manual_labels/highlights.json",
    "eval/manual-labels_v2/moments.csv",
    "calibration-0123456789abcdef/out/tracks.jsonl",
    "_scratch-0123456789abcdef/job.json",
    # Third review: .gitignore hid these names with a few data suffixes only, so git add
    # -A picked up other formats, compressed copies and backups that the gate flags.
    "plugin/detections.tsv",
    "plugin/detections.parquet",
    "plugin/detections.json5",
    "plugin/detections.jsonc",
    "plugin/detections.geojson",
    "plugin/detections.xml",
    "plugin/detections.npz",
    "plugin/ball_raw.h5",
    "plugin/ball_raw.feather",
    "plugin/ball_raw.jsonl.gz",
    "plugin/ball-raw.jsonl.zst",
    "plugin/detections.csv.bz2",
    "plugin/detections.parquet.xz",
    "plugin/detections.jsonl.bak",
    "plugin/detections.jsonl~",
    "plugin/detections.jsonl.orig",
    "plugin/ball_raw.ndjson.old",
    "plugin/detections.json.backup",
    "plugin/detections.jsonl.tmp",
    "plugin/detections.csv.save",
    "cfg/identity_overrides.json.bak",
    "cfg/identity_overrides.yaml~",
    "cfg/identity-overrides.jsonc",
    "cfg/identity_overrides.toml.gz",
]
#: Code, docs and JSON Schemas with the same names, and other ordinary files: .gitignore
#: shows them and the gate lets them through.
ORDINARY_FILES = [
    "src/demo/label_eval.py",
    "src/demo/identity_overrides.py",
    "src/demo/detections.py",
    "src/demo/ball_raw.pyi",
    "src/demo/detections/__init__.py",
    "docs/cases.md",
    "docs/detections.md",
    "docs/ball_raw.md",
    "docs/identity-overrides.rst",
    "docs/calibration.md",
    "src/gbf/contracts/schemas/plugin-receipt.schema.json",
    "configs/detector.yaml",
    "tests/fixtures/receipt.json",
    # Code and docs that name a data format, a compressed copy or a backup before their
    # own suffix.
    "docs/detections.jsonl.md",
    "docs/ball_raw.parquet.rst",
    "src/demo/detections_gz.py",
    "src/demo/identity_overrides.bak.pyi",
]


def test_gitignore_and_the_gate_agree_on_restricted_names(gate_repo: Path) -> None:
    # Second review: .gitignore hid copies such as detections (1).jsonl that the gate let
    # through when they were force-added. Both follow the same names now.
    shutil.copyfile(repo_root() / ".gitignore", gate_repo / ".gitignore")
    # Where they differ, the gate is the stricter one: .gitignore names data, compressed
    # and backup suffixes, so that it needs no negation, and the gate flags any suffix but
    # code and docs: a figure, a name without a suffix, and (fourth review, as the docs
    # now say) archives, numbered or partial copies, editor and lock files, spreadsheets
    # and databases.
    flagged_only = [
        "docs/figures/detections-overview.svg",
        "cfg/match7_identity-overrides",
        "plugin/detections.zip",
        "plugin/detections.br",
        "plugin/detections.jsonl.1",
        "plugin/detections.part",
        "plugin/.detections.jsonl.swp",
        "plugin/.~lock.detections.csv#",
        "plugin/detections.xlsx",
        "plugin/detections.sqlite",
    ]
    for rel in RESTRICTED_VARIANTS + ORDINARY_FILES + flagged_only:
        write_text(gate_repo / rel, "{}\n")
    # A JSON Schema elsewhere passes the gate but stays hidden until .gitignore gets a
    # negation for it, and so does one in the schema directory that is named like
    # restricted material: third review, .gitignore re-includes only the plugin receipt's
    # schema there, and fourth review, only plugin-receipt.schema.json itself.
    hidden_only = [
        "src/demo/schemas/detections.schema.json",
        "src/gbf/contracts/schemas/detections.schema.json",
        "src/gbf/contracts/schemas/plugin-receipt.v2.schema.json",
    ]
    for rel in hidden_only:
        write_json(gate_repo / rel, JSON_SCHEMA)
    # A directory named like such material with a backup suffix is hidden with the code
    # below it, which the gate lets through (fourth review, now in the docs).
    hidden_code = ["src/demo/detections.old/__init__.py"]
    for rel in hidden_code:
        write_text(gate_repo / rel, '"""Old detections."""\n')
    every = RESTRICTED_VARIANTS + ORDINARY_FILES + flagged_only + hidden_only + hidden_code
    files = set(candidate_files(gate_repo))
    result = check_restricted_data(gate_repo, Workspace(gate_repo, files=every))
    assert sorted(rel for rel in every if rel not in files) == sorted(
        RESTRICTED_VARIANTS + hidden_only + hidden_code
    )
    assert sorted({p.path for p in result.problems}) == sorted(RESTRICTED_VARIANTS + flagged_only)


def _ignored_by_git(root: Path, rels: list[str], *options: str) -> list[str]:
    """The paths among ``rels`` that ``git check-ignore`` reports as ignored, sorted."""
    completed = subprocess.run(
        ["git", "check-ignore", *options, "-z", "--stdin"],
        cwd=root,
        input="\0".join(rels).encode("utf-8"),
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode in (0, 1), completed.stderr  # 1: none of them is ignored
    return sorted(raw.decode("utf-8") for raw in completed.stdout.split(b"\0") if raw)


def test_gitignore_leaves_a_developers_own_excludes_alone(
    gate_repo: Path, unicode_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Second review: negations such as !*.py and !*.md at the end of .gitignore re-included
    # every such file that a developer excluded in .git/info/exclude or a global excludes
    # file, since .gitignore takes precedence over both, and git add -A picked them up.
    shutil.copyfile(repo_root() / ".gitignore", gate_repo / ".gitignore")
    write_text(
        gate_repo / ".git" / "info" / "exclude",
        "notes.md\nprobe.py\n*_scratch.*\nsrc/gbf/contracts/schemas/draft.schema.json\n"
        "src/gbf/contracts/schemas/plugin-receipt.v2.schema.json\n",
    )
    personal = unicode_tmp_path / "personal-excludes"
    write_text(personal, "todo.md\nlocal_experiment.py\n")
    config = unicode_tmp_path / "global.gitconfig"
    write_text(config, f"[core]\n\texcludesFile = {personal.as_posix()}\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    private = [
        "notes.md",
        "src/demo/probe.py",
        "docs/todo.md",
        "tools/local_experiment.py",
        "src/demo/detections_scratch.py",
        "docs/identity_overrides_scratch.md",
        # Third review: the negation for the project's JSON Schemas re-included every
        # schema in their directory, also the ones a developer excluded.
        "src/gbf/contracts/schemas/draft.schema.json",
        "src/gbf/contracts/schemas/receipt_scratch.schema.json",
        # Fourth review: it still re-included a schema there named like the plugin
        # receipt's, such as a later version of it, also when a developer excluded it.
        "src/gbf/contracts/schemas/plugin-receipt.v2.schema.json",
    ]
    for rel in private:
        write_text(gate_repo / rel, "private\n")
    assert _ignored_by_git(gate_repo, private) == sorted(private)
    assert not set(private) & set(candidate_files(gate_repo))


def test_gitignore_hides_no_file_the_repository_tracks(gate_repo: Path) -> None:
    # The patterns for restricted names must not hide the project's own files, such as
    # plugin-receipt.schema.json, which the one negation re-includes. git lists them, so
    # tests run from a source export, which is no git work tree, cannot check this (fourth
    # review: git ls-files failed there with an error that did not say why). The gate
    # itself runs in a work tree, where this test never skips.
    if not (repo_root() / ".git").exists():
        pytest.skip("not run from a git work tree, so git cannot list the tracked files")
    shutil.copyfile(repo_root() / ".gitignore", gate_repo / ".gitignore")
    listed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=repo_root(), capture_output=True, check=True, timeout=60
    )
    tracked = [raw.decode("utf-8") for raw in listed.stdout.split(b"\0") if raw]
    assert "src/gbf/contracts/schemas/plugin-receipt.schema.json" in tracked
    assert _ignored_by_git(gate_repo, tracked, "--no-index") == []


# --- forbidden imports -----------------------------------------------------------------
def test_forbidden_imports_in_every_form(gate_repo: Path) -> None:
    write_text(
        gate_repo / "src" / "demo" / "plugin_bridge.py",
        "from sn_gamestate.calibration import nbjw\n"
        "import os, tracklab.utils\n"
        "import importlib\n"
        "mod = importlib.import_module('mmcv.ops')\n"
        "loader = __import__('pickle')\n"
        "from . import pickle_utils\n"
        "from .pickle import helpers\n"
        "print(os, nbjw, mod, loader, pickle_utils, helpers, tracklab)\n",
    )
    write_text(gate_repo / "tests" / "test_dill.py", "import dill\n")
    write_text(gate_repo / "plugins" / "sngs" / "run.py", "import tracklab\n")
    write_text(gate_repo / "src" / "demo" / "broken.py", "def f(:\n")
    assert lines(check_forbidden_imports(gate_repo)) == [
        "forbidden_imports src/demo/broken.py:1 syntax-error",
        "forbidden_imports src/demo/plugin_bridge.py:1 sn_gamestate",
        "forbidden_imports src/demo/plugin_bridge.py:2 tracklab",
        "forbidden_imports src/demo/plugin_bridge.py:4 mmcv",
        "forbidden_imports src/demo/plugin_bridge.py:5 pickle",
        "forbidden_imports tests/test_dill.py:1 dill",
    ]


FIND_SPEC_THEN_LOAD = """\
import importlib.util

spec = importlib.util.find_spec("ultralytics")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
"""


@pytest.mark.parametrize(
    ("rel", "source", "line"),
    [
        pytest.param(
            "src/demo/loader.py",
            'from importlib import import_module as im\n\nim("ultralytics")\n',
            3,
            id="aliased-import-module",
        ),
        pytest.param(
            "src/demo/loader.py",
            'import importlib\n\nimportlib.import_module(name="ultralytics.engine")\n',
            3,
            id="import-module-keyword",
        ),
        pytest.param(
            "src/demo/loader.py", 'yolo = __import__(name="ultralytics")\n', 1, id="dunder-keyword"
        ),
        pytest.param(
            "tests/test_loader.py",
            'import pytest\n\nyolo = pytest.importorskip("ultralytics")\n',
            3,
            id="importorskip",
        ),
        pytest.param("src/demo/loader.py", FIND_SPEC_THEN_LOAD, 3, id="find-spec-then-loader"),
        # Further spellings of the same calls.
        pytest.param(
            "src/demo/loader.py",
            "import importlib\n\nload = importlib.import_module\nfetch = load\n"
            "fetch('ultralytics')\n",
            5,
            id="assigned-alias-chain",
        ),
        pytest.param(
            "src/demo/loader.py",
            "from importlib.util import find_spec as locate\n\nlocate(fullname='ultralytics')\n",
            3,
            id="aliased-find-spec-keyword",
        ),
        pytest.param(
            "tests/test_loader.py",
            "from pytest import importorskip as needs\n\nneeds(modname=f'ultralytics')\n",
            3,
            id="aliased-importorskip-fstring",
        ),
        pytest.param(
            "src/demo/loader.py",
            "from builtins import __import__ as imp\n\nimp('ultra' + 'lytics')\n",
            3,
            id="aliased-dunder-concatenated",
        ),
        pytest.param(
            "src/demo/loader.py",
            "import importlib\n\nimportlib.import_module('.engine', package='ultralytics')\n",
            3,
            id="relative-name-and-package",
        ),
        pytest.param(
            "src/demo/loader.py",
            "import runpy\n\nrunpy.run_module(mod_name='ultralytics')\n",
            3,
            id="run-module",
        ),
        pytest.param(
            "src/demo/loader.py",
            "import pkgutil\n\nYOLO = pkgutil.resolve_name('ultralytics:YOLO')\n",
            3,
            id="resolve-name",
        ),
        # Second review: loaders, aliases and files the check did not see.
        pytest.param(
            "src/demo/loader.py",
            "import pkgutil\n\nloader = pkgutil.get_loader('ultralytics')\n",
            3,
            id="get-loader",
        ),
        pytest.param(
            "src/demo/loader.py",
            "import pkgutil\n\nloader = pkgutil.find_loader(fullname='ultralytics')\n",
            3,
            id="find-loader",
        ),
        pytest.param(
            "src/demo/loader.py",
            "import importlib\n\nload, count = importlib.import_module, 1\nload('ultralytics')\n",
            4,
            id="alias-unpacked-from-a-tuple",
        ),
        pytest.param(
            "src/demo/loader.py",
            "import importlib\n\n[(n, load)] = [(1, importlib.import_module)]\n"
            "load('ultralytics')\n",
            4,
            id="alias-unpacked-from-nested-sequences",
        ),
        pytest.param(
            "src/demo/loader.py",
            "import importlib\n\n(load := importlib.import_module)('ultralytics')\n",
            3,
            id="alias-bound-and-called-at-once",
        ),
        pytest.param("src/demo/loader.pyw", "import ultralytics\n", 1, id="pyw-script"),
        pytest.param(
            "tests/test_loader.pyw",
            "import pkgutil\n\npkgutil.get_loader(module_or_name='ultralytics')\n",
            3,
            id="pyw-test-get-loader-keyword",
        ),
    ],
)
def test_loader_calls_with_a_literal_name_are_imports(
    gate_repo: Path, rel: str, source: str, line: int
) -> None:
    write_text(gate_repo / rel, source)
    assert lines(check_forbidden_imports(gate_repo)) == [
        f"forbidden_imports {rel}:{line} ultralytics"
    ]


def test_loader_calls_with_computed_names_or_allowed_modules_pass(gate_repo: Path) -> None:
    # Computed names are out of scope (gbf.gate.imports); a test that checks that a banned
    # module is absent builds its name at run time, as here.
    write_text(
        gate_repo / "tests" / "test_environment.py",
        "import importlib\nimport importlib.util\nimport pkgutil\n\nimport pytest\n\n"
        "BANNED = ''.join(['ultra', 'lytics'])\n\n\n"
        "def test_core_environment() -> None:\n"
        "    assert importlib.util.find_spec(BANNED) is None\n"
        "    assert importlib.util.find_spec('json') is not None\n"
        "    assert pytest.importorskip('json')\n"
        "    assert importlib.import_module('.decoder', package='json')\n"
        "    assert pkgutil.get_loader(BANNED) is None\n"
        "    load, name = importlib.import_module, BANNED\n"
        "    assert load('json') and name\n",
    )
    assert check_forbidden_imports(gate_repo).passed


def test_names_of_blocked_registry_entries_are_banned(gate_repo: Path) -> None:
    # rf-detr-plus and deimv2 are blocked in registries/models.toml.
    write_text(
        gate_repo / "src" / "demo" / "detector.py",
        "import rfdetr_plus\nfrom deimv2.engine import core\nprint(rfdetr_plus, core)\n",
    )
    text = (gate_repo / "pyproject.toml").read_text(encoding="utf-8")
    text = text.replace('"numpy>=2.0",', '"numpy>=2.0",\n    "rf-detr-plus>=1.0",\n    "DEIMv2",')
    write_text(gate_repo / "pyproject.toml", text)
    assert lines(check_forbidden_imports(gate_repo)) == [
        "forbidden_imports src/demo/detector.py:1 rfdetr_plus",
        "forbidden_imports src/demo/detector.py:2 deimv2",
    ]
    assert lines(check_deps(gate_repo)) == [
        f"deps pyproject.toml:{_line_of(text, 'rf-detr-plus')} rf-detr-plus (project.dependencies)",
        f"deps pyproject.toml:{_line_of(text, 'DEIMv2')} deimv2 (project.dependencies)",
    ]


MODELS_REGISTRY = """\
schema_version = 1

[[models]]
id = "boxmot"
status = "blocked"

[[models]]
id = "fresh-detector"
status = "planned"

[[models]]
id = "new-tracker"
status = "blocked"

[[models]]
id = "new-research-model"
status = "research_only"
"""


def test_a_banned_registry_entry_missing_from_the_ban_table_fails(gate_repo: Path) -> None:
    write_text(gate_repo / "registries" / "models.toml", MODELS_REGISTRY)
    text = MODELS_REGISTRY
    assert lines(check_forbidden_imports(gate_repo)) == [
        f"forbidden_imports registries/models.toml:{_line_of(text, 'new-tracker')} "
        "unmapped-registry-entry (new-tracker)",
        f"forbidden_imports registries/models.toml:{_line_of(text, 'new-research-model')} "
        "unmapped-registry-entry (new-research-model)",
    ]


def test_every_banned_entry_of_the_model_registry_has_a_ban_row() -> None:
    data = tomllib.loads((repo_root() / "registries" / "models.toml").read_text("utf-8"))
    statuses = imports_module.BANNED_STATUSES
    banned = {entry["id"] for entry in data["models"] if entry["status"] in statuses}
    bans = imports_module.REGISTRY_BANS
    assert imports_module.unmapped_registry_ids(data) == []
    assert sorted(set(bans) - banned) == []  # no row for an entry that is no longer banned
    empty = [key for key, ban in bans.items() if not (ban.imports or ban.distributions)]
    assert [key for key in empty if not bans[key].note] == []  # an empty row says why
    forbidden = imports_module.FORBIDDEN_MODULES
    assert {"ultralytics", "sn_gamestate", "rfdetr_plus", "deimv2", "pickle"} <= forbidden
    denied = imports_module.DENIED_DISTRIBUTIONS
    assert {"rf-detr-plus", "rfdetr-plus", "deimv2", "python-bidi"} <= denied


# --- deps ------------------------------------------------------------------------------
DENIED_PYPROJECT = """\
[project]
name = "gate-fixture"
version = "0.0.0"
dependencies = [
    "mylib @ git+https://example.invalid/ultralytics/ultralytics.git@v8",
]

[project.optional-dependencies]
viz = ["easyocr>=1.7"]

[dependency-groups]
research = ["Sn_GameState==1.0"]

[build-system]
requires = ["hatchling", "python_bidi"]

[tool.uv.sources]
tracklab = { git = "https://example.invalid/tracklab.git" }
"""


def test_denied_distributions_anywhere_in_pyproject(gate_repo: Path) -> None:
    write_text(gate_repo / "pyproject.toml", DENIED_PYPROJECT)
    result = check_deps(gate_repo)
    text = DENIED_PYPROJECT
    assert lines(result) == [
        f"deps pyproject.toml:{_line_of(text, 'mylib')} ultralytics (project.dependencies)",
        f"deps pyproject.toml:{_line_of(text, 'easyocr')} easyocr "
        "(project.optional-dependencies.viz)",
        f"deps pyproject.toml:{_line_of(text, 'Sn_GameState')} sn-gamestate "
        "(dependency-groups.research)",
        f"deps pyproject.toml:{_line_of(text, 'python_bidi')} python-bidi (build-system.requires)",
        f"deps pyproject.toml:{_line_of(text, 'tracklab = ')} tracklab (tool.uv.sources)",
    ]
    assert any("P5" in note for note in result.notes)


def test_requirement_names_cover_extras_markers_and_archive_urls() -> None:
    assert requirement_names("Ultralytics[export]>=8; python_version>'3.10'") >= {"ultralytics"}
    archive = "x @ https://files.example.invalid/boxmot-10.0.71-py3-none-any.whl"
    assert "boxmot" in requirement_names(archive)
    assert requirement_names("numpy>=2.0") == {"numpy"}


def test_deps_needs_a_readable_pyproject(gate_repo: Path) -> None:
    (gate_repo / "pyproject.toml").unlink()
    write_text(gate_repo / "uv.lock", 'version = 1\n[[package]\nname = "x"\n')
    assert lines(check_deps(gate_repo)) == [
        "deps pyproject.toml missing",
        "deps uv.lock:2 toml-parse-error",
    ]


# --- registries ------------------------------------------------------------------------
def test_required_registries_come_from_gate_config(gate_repo: Path) -> None:
    write_json(gate_repo / "gate" / "registries.json", {"schema_version": 1, "required": ["tools"]})
    assert lines(check_registries(gate_repo)) == ["registries registries/tools.toml missing"]


def test_gate_registry_config_lists_every_problem(gate_repo: Path) -> None:
    write_json(
        gate_repo / "gate" / "registries.json",
        {"schema_version": 2, "required": ["Media!", "tools", "tools"]},
    )
    assert lines(check_registries(gate_repo)) == [
        "registries gate/registries.json invalid (required holds an invalid kind)",
        "registries gate/registries.json invalid (required must list each kind once)",
        "registries gate/registries.json invalid (schema_version must be 1)",
        "registries registries/tools.toml missing",
    ]


def test_a_registry_without_a_schema_is_reported_missing(gate_repo: Path) -> None:
    write_text(gate_repo / "registries" / "gatetestonly.toml", "schema_version = 1\n")
    assert lines(check_registries(gate_repo)) == [
        "registries registries/gatetestonly.toml schema-missing "
        "(urn:gbf:schema:registry_gatetestonly)"
    ]


def test_registries_are_validated_against_their_schema(gate_repo: Path) -> None:
    write_text(gate_repo / "registries" / "media.toml", "schema_version = 2\n")
    problems = lines(check_registries(gate_repo))
    assert "registries registries/media.toml schema-const (/schema_version)" in problems
    assert "registries registries/media.toml schema-required (/media)" in problems


def test_registry_parse_errors_carry_the_line(gate_repo: Path) -> None:
    write_text(gate_repo / "registries" / "media.toml", "schema_version = 1\nmedia = [\n  {\n")
    result = check_registries(gate_repo)
    assert [(p.rule, p.line is not None) for p in result.problems] == [("toml-parse-error", True)]


def test_registered_repository_files_are_rehashed_over_raw_bytes(gate_repo: Path) -> None:
    fixtures = gate_repo / "tests" / "fixtures"
    fixtures.mkdir(parents=True)
    (fixtures / "a.txt").write_bytes(b"a\n")
    (fixtures / "b.txt").write_bytes(b"b\n")
    (fixtures / "c.txt").write_bytes(b"c\n")
    good = sha256_bytes(b"a\n")
    wrong = sha256_bytes(b"other")
    zeros = "0" * 64
    registry = (
        "schema_version = 1\n\n"
        f'[[files]]\npath = "./tests/fixtures/a.txt"\nsha256 = "{good}"\n\n'
        '[[files]]\nrelpath = "tests/fixtures/c.txt"\nsha256 = "XYZ"\n\n'
        f'[[files]]\npath = "elsewhere/not-in-repo.bin"\nsha256 = "{zeros}"\n\n'
        f'[by_path."tests/fixtures/b.txt"]\nsha256 = "{wrong}"\n'
    )
    write_text(gate_repo / "registries" / "gatetestonly.toml", registry)
    problems = [p for p in lines(check_registries(gate_repo)) if "schema-missing" not in p]
    assert problems == [
        "registries tests/fixtures/b.txt sha256-mismatch "
        "(registered in registries/gatetestonly.toml)",
        "registries tests/fixtures/c.txt sha256-malformed "
        "(registered in registries/gatetestonly.toml)",
    ]


def test_registry_entries_and_toml_dates() -> None:
    data = {
        "media": [{"path": "x.png", "sha256": "ab", "restrictions": ["nda", 3]}],
        "files": {"docs/y.svg": {"sha256": "cd"}},
        "tools": [{"name": "ffmpeg"}],
    }
    entries = registry_entries(data)
    assert [(e.path, e.sha256, e.restrictions, e.pointer) for e in entries] == [
        ("x.png", "ab", ("nda",), "/media/0"),
        ("docs/y.svg", "cd", (), "/files/docs/y.svg"),
    ]
    assert to_json_compatible({"d": date(2026, 10, 21), "l": [date(2026, 1, 2)]}) == {
        "d": "2026-10-21",
        "l": ["2026-01-02"],
    }


def _models_registry(**fields: object) -> dict[str, object]:
    """A registries/models.toml document with one synthetic experimental entry."""
    entry: dict[str, object] = {
        "id": "demo-pose",
        "name": "Synthetic pose model for the schema tests",
        "role": "pose_2d",
        "upstream_url": "https://example.invalid/demo-pose",
        "code_license": "Apache-2.0",
        "weights_license": "NOASSERTION",
        "training_data": "Not verified.",
        "restrictions": [],
        "status": "experimental",
        "allowed_in_core": False,
        "default_enabled": False,
        "plugin": "perception-extra",
        "verified_url": "https://example.invalid/demo-pose",
        "verified_on": "2026-10-21",
        "notes": "Synthetic.",
    }
    entry.update(fields)
    return {"schema_version": 1, "models": [entry]}


CORE_EXCEPTION = "ADR 0001 amendment of 2026-10-21."


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        (  # the core runs it under a recorded ADR 0001 exception, with research_nc
            {
                "allowed_in_core": True,
                "default_enabled": True,
                "restrictions": ["research_nc"],
                "plugin": "none",
                "core_exception": CORE_EXCEPTION,
            },
            [],
        ),
        ({"restrictions": ["research_nc"]}, []),  # research_nc outside the core
        (  # in the core without a recorded exception
            {"allowed_in_core": True, "default_enabled": True, "restrictions": ["research_nc"]},
            [
                ("const", "/models/0/allowed_in_core"),
                ("const", "/models/0/default_enabled"),
                ("const", "/models/0/status"),
                ("maxItems", "/models/0/restrictions"),
            ],
        ),
        ({"core_exception": CORE_EXCEPTION}, [("const", "/models/0/allowed_in_core")]),
        (
            {"status": "planned", "allowed_in_core": True, "core_exception": CORE_EXCEPTION},
            [("const", "/models/0/status")],
        ),
        ({"status": "planned", "restrictions": ["research_nc"]}, [("enum", "/models/0/status")]),
    ],
)
def test_model_registry_schema_allows_core_entries_with_restrictions_only_as_exceptions(
    fields: dict[str, object], expected: list[tuple[str, str]]
) -> None:
    assert schema_problems(_models_registry(**fields), "registry_models") == expected


def test_the_model_registry_records_the_pose_models_the_core_runs() -> None:
    data = tomllib.loads((repo_root() / "registries" / "models.toml").read_text("utf-8"))
    assert schema_problems(to_json_compatible(data), "registry_models") == []
    pinned = {pin.model_id for pin in PINS}
    (entry,) = [e for e in data["models"] if pinned <= set(e.get("variants", []))]
    assert (entry["allowed_in_core"], entry["default_enabled"]) == (True, True)
    assert "research_nc" in entry["restrictions"]
    assert "ADR 0001" in entry["core_exception"]
    (rtmw,) = [e for e in data["models"] if e["id"] == "rtmw-l"]  # same family and data
    assert "research_nc" in rtmw["restrictions"]


# --- receipts --------------------------------------------------------------------------
def test_receipt_parse_errors_and_non_standard_numbers(gate_repo: Path) -> None:
    public = gate_repo / "receipts" / "public"
    write_text(public / "broken.json", '{\n  "kind":\n')
    write_text(public / "nan.json", '{"x": NaN}\n')
    (public / "latin1.json").write_bytes(b'{"x": "\xe9"}\n')
    assert lines(check_receipts(gate_repo)) == [
        "receipts receipts/public/broken.json:3 json-parse-error",
        "receipts receipts/public/latin1.json not-utf8",
        "receipts receipts/public/nan.json json-nonstandard-number",
    ]


def test_receipt_schema_problems_never_echo_instance_keys(
    valid_receipt: dict[str, object], make_token: Callable[..., str]
) -> None:
    odd_key = "weird key: " + make_token("", 12)
    receipt = dict(valid_receipt, environment={odd_key: 5, "ffmpeg": "8.1.1"})
    assert schema_problems(receipt, "receipt") == [("type", "/environment/?")]


def test_valid_receipts_pass(gate_repo: Path, valid_receipt: dict[str, object]) -> None:
    write_json(gate_repo / "receipts" / "public" / "nested" / "ok.json", valid_receipt)
    result = check_receipts(gate_repo)
    assert result.passed
    assert result.scanned == 2


def test_abs_paths_check_reports_each_line_once(
    gate_repo: Path, make_token: Callable[..., str]
) -> None:
    name = make_token("gbfuser", 5, string.ascii_lowercase)
    text = f"/home/{name}/a and C:/Users/{name}/b\nok\n/Users/{name}/c\n"
    write_text(gate_repo / "docs" / "paths.txt", text)
    assert lines(check_abs_paths(gate_repo)) == [
        "abs_paths docs/paths.txt:1 user-home-path",
        "abs_paths docs/paths.txt:3 user-home-path",
    ]
