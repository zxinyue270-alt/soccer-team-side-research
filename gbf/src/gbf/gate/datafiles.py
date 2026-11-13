"""Checks that keep media and restricted data out of the repository (ADR 0002, gate G4).

media: a candidate file with a media, data, model-weight or archive suffix (see
``gbf.gate.repo.MEDIA_SUFFIXES``) must be registered in ``registries/media.toml``
(an entry naming its path, see ``gbf.gate.registries.registry_entries``) with the sha256
of its raw bytes, and the entry's ``restrictions`` must not contain nda, no_share,
research_nc or no_identity.

restricted_data always fails, registered or not, on:

- StatsBomb-shaped files: ``events|lineups|three-sixty/<digits>.json`` or
  ``matches/[<digits>/]<digits>.json`` anywhere in the tree, and JSON (.json, .jsonl,
  .ndjson, .geojson) whose first 64 KiB contains both possession_team and play_pattern;
- SoccerNet label files (``Labels-*.json``);
- anything below a directory named ``runs`` or ``data`` (as in .gitignore, at any depth);
- pickle-family files (.pkl, .pickle, .pklz, .joblib, .dill, also inside a double
  suffix such as .pkl.gz);
- evidence stores: data-store files (.jsonl, .ndjson, .parquet, .sqlite, .sqlite3, .db,
  .duckdb, .arrow, .feather, .npz, .h5, .hdf5, .lmdb) named evidence* or placed below a
  directory named evidence, evidence_store or evidence-store;
- NDA case files: anything below a directory named ``cases`` (``nda-case-file``);
- manual labels: anything below a directory named ``labels``, ``label`` or
  ``manual_labels`` (also ``manual-labels``, ``manual_label`` and ``manual_labels_v2``
  and the like; ``under-labels``), JSON or JSON lines (.json, .jsonl, .ndjson, .ldjson)
  named ``manual_*`` or ``manual-*`` anywhere (``manual-labels``), and hand labels under
  any name (``manual-labels-content``): a JSON document, read where the restricted-tag
  rule below reads tags, with an object whose ``moments`` list holds, among its first
  JSON_ARRAY_SNIFF_ELEMENTS items, an object with the keys gbf.label_eval needs
  (HAND_LABEL_MOMENT_KEYS: id, type, start, peak and end);
- plugin job directories: anything below a directory named ``<task>-<16 hex digits>``,
  the name gbf.plugins.protocol.job_dir_name gives them (with any name before the
  hyphen, as in .gitignore): the job.json, the copies of the inputs under in/, the raw
  outputs under out/ and the plugin receipt (``under-plugin-job``);
- identity overrides: a name that contains ``identity_override`` or
  ``identity-override``, with any extension (``identity-overrides``):
  identity_overrides.json, identity_overrides (1).json, match7_identity-overrides.yaml;
- raw plugin outputs (``raw-plugin-output``): the names in RAW_PLUGIN_OUTPUTS (the
  plugin-sngs export detections.jsonl and calib.jsonl, the ball candidates
  ball_raw.jsonl and their ball-receipt.json, the pitch keypoints keypoints.jsonl and
  keypoints-receipt.json, and the protocol's plugin-receipt.json) and renamed copies
  (RAW_PLUGIN_OUTPUT_PATTERNS): a name that contains ``detections``, ``ball_raw`` or
  ``ball-raw``, with any extension (detections_v2.jsonl, detections (1).jsonl,
  Copy of detections.jsonl, detectionsV2.json, ball_raw2.ndjson), JSON lines (.jsonl,
  .ndjson, .ldjson) whose name starts with ``calib`` or ``keypoints`` (calib (1).jsonl,
  keypoints_v2.ndjson), and JSON whose name contains ``plugin-receipt``,
  ``keypoints-receipt`` or ``ball-receipt``, also with an underscore (ball-receipt
  (1).json, plugin-receipt.v2.json);
- restricted content (``restricted-tag``, one problem per tag): a JSON document (a .json
  or .geojson file, a .json5 or .jsonc file that parses as JSON, or one record of a
  .jsonl, .ndjson or .ldjson file) with nda, no_share, research_nc or no_identity under
  a ``restrictions`` key, in any letter case, of the top-level object, of an object one
  level down (``{"run": {"restrictions": [...]}}``) or of one of the first
  JSON_ARRAY_SNIFF_ELEMENTS elements of a top-level array, each read like a top-level
  object; the value is a list of tags or a string that holds them (``"no_share, nda"``);
  a JSON lines file is judged by its first JSON_LINES_SNIFF_RECORDS records. A page
  (.html, .htm, .shtml, .xhtml, .xht, .svg, .xml, and a saved page, .mht or .mhtml, read
  as it is and part by part, decoded as its headers declare, GB18030_CHARSETS as gb18030
  like a browser) fails when it states such a tag as a ``restrictions`` list or
  attribute, or in the string after a quoted restrictions key (``"restrictions":
  "no_share"``, as JSON in a page or a .jsonc file writes it), each up to the bracket or
  quote that closes it: the quote of the kind that opened it, a backslash escaping the
  next character (``"partner's nda"`` and ``"hl3 \\"x\\" nda"`` are read whole), and
  after an attribute's ``=`` also without escapes, as HTML reads it
  (``data-restrictions="nda\\"``); in the content or value attribute of an element named
  restrictions (by its tag name, name, property, itemprop or http-equiv:
  ``<meta name="restrictions" content="...">``), or in the text of an element named so,
  also by its class or id, that holds nothing but tags
  (``<ul class="restrictions"><li>no_share</li></ul>``,
  ``<restrictions>nda</restrictions>``); in a table or definition list one of whose
  cells reads "restrictions", "restriction tags" or 限制标签; after a ``Restrictions:`` or
  ``Restriction tags:`` label, up to the end of its line, element or sentence and on
  through further ``;``-separated lists of tags; in the 限制标签 list of gbf's reports; on
  the lines below such a label that fills its line, as long as they hold nothing but
  tags (``<p>Restrictions:</p><ul><li>no_share</li></ul>``); and in SVG text elements
  that hold nothing but tags and follow one that reads "Restrictions" or 限制标签, with or
  without a colon. "Nothing but tags" means tags of
  src/gbf/contracts/schemas/restrictions.schema.json and separators. Entities are
  decoded, what a browser does not draw (zero-width spaces, soft hyphens, ``<wbr>``) is
  dropped, also inside a label word (``Re<wbr>strictions:``), and inline markup does not
  split a statement (it does split a word, see below). A page that carries the
  private-material banner 私有素材, also split by markup or by such characters, fails as
  ``private-banner``. A .json5 or .jsonc file that does not parse as JSON is read like
  the text of a page, without the banner rule. A backup copy is read like the file it
  copies (report.html.bak, report.html~, rows.jsonl.orig);
- content that cannot be scanned, which fails closed: a JSON document, page or saved page
  larger than TEXT_SCAN_LIMIT, or a JSON lines file whose first JSON_LINES_SNIFF_RECORDS
  records do not fit in its first JSON_LINES_SNIFF_BYTES (``too-large-to-scan``); bytes
  that are not UTF-8, UTF-16 or UTF-32, or a part of a saved page that is not in its
  declared charset either (``undecodable``); a .json or .geojson document that does not
  parse while its text mentions restrictions (``json-parse-error``); and a saved page
  whose parts are nested too deeply for Python's email parser, whose MIME header
  parameters cannot be parsed (including malformed RFC 2231 continuations), or with a text part that
  declares a charset Python cannot use at all, such as a name with a NUL, also in the
  RFC 2231 form ``charset*=``, or a codec that fails otherwise (``mime-parse-error``).
  That part is still read, as UTF-8, UTF-16 or UTF-32; a part whose charset Python does
  not know is read so too, without a problem.

The name rules skip code and documentation (NAME_RULE_EXEMPT_SUFFIXES) and JSON Schemas:
a ``.schema.json`` file under JSON_SCHEMA_DIR, or one whose top level holds a
``"$schema"`` string. So a module may be called identity_overrides.py and a schema
detections.schema.json, while a data file named like a schema is still caught.

Not covered: tags deeper in a JSON document (``{"items": [{"restrictions": [...]}]}``,
which also lets JSON Schemas with their enums of tags pass), tags under another key
(``restriction_tags``), a ``restrictions`` value that is an object, duplicate keys (the
parser keeps the last), a JSON document encoded as a string inside another, records
after the ones sniffed, tags spelled otherwise (``no-share``, full-width letters, a
zero-width space inside a JSON string), ``\\u`` escapes in JSON that is read as text (in
a page, or in a .json5 or .jsonc file that does not parse: ``"restr\\u0069ctions"``,
``"\\u006eda"``), hand labels in another shape (a moments list nested deeper, the
moments without one, keys spelled otherwise) or in a .json5 or .jsonc file that does not
parse, and restriction statements in other wording or shapes: a label inside a sentence
("these restrictions apply: nda", "the gate blocks these restrictions:" above a list), a
heading without a colon above a list, a dash or an unquoted ``=`` in place of the colon
(``Restrictions — nda``, ``Restrictions = nda``), a label and its tags in separate
tspans of one SVG text element, lines or elements below a label that hold other words or
punctuation besides the tags (``nda, no_share.``), more than TAG_LISTS_LIMIT of them, a
list or string after a restrictions key or attribute that does not close within 512
characters, a list that holds ``]``, a string that holds ``<`` or ``>``, tags more than
512 characters after a label, a tag or a label word split by markup other than
``<wbr>``, a comment included (``no_<b>share</b>``, ``<b>Restric</b>tions: nda``,
``Re<span>strictions</span>:``, ``限制<b>标签</b>：``), an element named restrictions by
another attribute (``aria-label``, ``<prop key="restrictions" value="nda"/>``), tags
after an element that is nested in one named restrictions and has the same tag name (the
text ends at the first end tag of that name), tags in a CDATA section, tags in an
element that does not close within ELEMENT_TEXT_LIMIT characters, and tags only an
image, a style sheet or a script would draw. The content of other formats is not read
(YAML, TOML, CSV, text, Markdown, Office files, PDFs, archives, pages under other
suffixes such as .eml, .php or .vue; registries are TOML and are not read here), nor a
part of a saved page that is not text (``application/octet-stream``), nor a backup with
another suffix (report.html.1, a .swp file). Material saved under other names or in
other directories passes the name rules (detection.jsonl, ballraw.jsonl, ball raw.jsonl,
IdentityOverrides.json, identity overrides.json, calib.json, Copy of calib.jsonl,
match7_keypoints.jsonl, manual_labels.csv, labels_v2/, manual_label_v2/, nda_cases/,
out/ of a job directory that was renamed), and so do backups of the names restricted
with certain suffixes only (plugin-receipt.json.bak, calib.jsonl~) and data saved under
a JSON Schema's name in JSON_SCHEMA_DIR or with a top-level ``"$schema"`` string.

False positives, the price of reading statements broadly: an ordinary page or string
fails when it reads like a statement, such as a glossary in a table or definition list
labelled restrictions whose other cells mention a tag, a list of tags below a heading
"Restriction tags:", a form whose select named restrictions offers tags, and a
``restrictions`` string in JSON that mentions a tag in prose ("none; see ADR 0002 for
nda handling", "No NDA material is used"). The name rules flag identity_override,
detections and ball_raw names with any suffix but NAME_RULE_EXEMPT_SUFFIXES, so a
figure, a notebook or a .pyw script named so fails too. A part of a saved page fails as
undecodable where Python decodes its charset more strictly than a browser (bytes that
windows-1252 leaves undefined).

Conventions that keep ordinary files out of these rules (CONTRIBUTING.md): the directory
rules apply to every file, so code packages and fixture directories are never named
runs, data, cases, labels, label or manual_labels; JSON Schemas live in JSON_SCHEMA_DIR;
synthetic plugin outputs and hand labels for tests are generated at test time; and a page
that explains a restriction statement writes a placeholder, not a blocked tag.

Names are compared case-insensitively. .gitignore hides the same names with the suffixes
such material is written with (names that contain identity_override, detections or
ball_raw as data, compressed copies and backups; see its comment), so they do not become
candidates by accident, and re-includes one file only, the plugin receipt's schema
plugin-receipt.schema.json in JSON_SCHEMA_DIR, so a developer's own excludes keep
working. Besides what is force-added, these rules catch the names .gitignore leaves
visible: those three with other suffixes, such as a figure, a spreadsheet or database
(.xlsx, .db, .sqlite, .duckdb), an archive or another compressor (.zip, .7z, .tar, .tgz,
.lz4, .lzma, .br, .zstd, .Z, .jsonl.zip), a numbered or partial copy (.jsonl.1, .bak1,
.bk, .part, .crdownload), an editor or lock file (.detections.jsonl.swp,
#detections.jsonl#, .~lock.detections.csv#) or a name without a suffix, and names in
another letter case where git matches .gitignore case-sensitively. The other way round,
.gitignore hides a directory named like such material with a backup suffix
(detections.old/, ball_raw.tmp/) and the code and docs below it, which these rules would
let through; Python packages cannot have such names. A JSON Schema named like restricted
material, other than plugin-receipt.schema.json itself (a later version of it is hidden
too), passes these rules but is hidden by .gitignore until it gets a negation of its own
there (CONTRIBUTING.md).
"""

from __future__ import annotations

import email
import html
import json
import quopri
import re
import tomllib
from functools import cache
from pathlib import Path
from typing import Any

from gbf import contracts
from gbf.gate.model import CheckResult
from gbf.gate.registries import RegistryEntry, registry_entries
from gbf.gate.repo import (
    MEDIA_SUFFIXES,
    TEXT_SCAN_LIMIT,
    Workspace,
    basename,
    suffix,
    suffixes,
)

MEDIA_REGISTRY = "registries/media.toml"
BLOCKED_RESTRICTIONS = ("nda", "no_share", "research_nc", "no_identity")
PICKLE_SUFFIXES = frozenset({".pkl", ".pickle", ".pklz", ".joblib", ".dill"})
STATSBOMB_SNIFF_BYTES = 64 * 1024

_STATSBOMB_PATH = re.compile(
    r"(?:^|/)(?:(?:events|lineups|three-sixty)/\d+|matches/(?:\d+/)?\d+)\.json$", re.IGNORECASE
)
_STATSBOMB_MARKERS = (b"possession_team", b"play_pattern")
_JSON_SUFFIXES = frozenset({".json", ".jsonl", ".ndjson", ".geojson"})
_SOCCERNET_LABELS = re.compile(r"Labels-.*\.json", re.IGNORECASE)
_RESTRICTED_DIRS = {
    "runs": "under-runs",
    "data": "under-data",
    "labels": "under-labels",
    "label": "under-labels",
    "cases": "nda-case-file",
}
_MANUAL_LABELS_DIR = re.compile(r"manual[-_]label(?:s.*)?")
#: A plugin job directory: <task>-<first 16 hex digits of the job sha256>, with any name
#: before the hyphen, as .gitignore has it.
_PLUGIN_JOB_DIR = re.compile(r".*-[0-9a-f]{16}", re.DOTALL)
#: Raw plugin outputs (docs/design/m1-highlights.md, ADR 0001 section 3), lower case.
RAW_PLUGIN_OUTPUTS = frozenset(
    {
        "detections.jsonl",
        "calib.jsonl",
        "ball_raw.jsonl",
        "ball-receipt.json",
        "keypoints.jsonl",
        "keypoints-receipt.json",
        "plugin-receipt.json",
    }
)
#: Renamed and copied raw plugin outputs (whole lower-case names, see the module
#: docstring); .gitignore hides the same names with data suffixes.
RAW_PLUGIN_OUTPUT_PATTERNS = (
    re.compile(r".*(?:detections|ball[-_]raw).*", re.DOTALL),
    re.compile(r"(?:calib|keypoints).*\.(?:jsonl|ndjson|ldjson)", re.DOTALL),
    re.compile(r".*(?:plugin|keypoints|ball)[-_]receipt.*\.json", re.DOTALL),
)
#: Code and documentation: the name rules (identity overrides, manual labels, raw plugin
#: outputs) do not apply to files ending in these.
NAME_RULE_EXEMPT_SUFFIXES = (".py", ".pyi", ".md", ".rst")
#: The project's JSON Schemas; the name rules skip a .schema.json file below it.
JSON_SCHEMA_DIR = "src/gbf/contracts/schemas/"
#: A JSON lines file is judged by this many records within its first
#: JSON_LINES_SNIFF_BYTES.
JSON_LINES_SNIFF_RECORDS = 32
JSON_LINES_SNIFF_BYTES = 1024 * 1024
#: A top-level JSON array, and the moments list of hand labels, are judged by this many
#: of their first elements.
JSON_ARRAY_SNIFF_ELEMENTS = 32
#: The keys gbf.label_eval needs in each labelled moment of a hand-label file.
HAND_LABEL_MOMENT_KEYS = frozenset({"id", "type", "start", "peak", "end"})
#: The text of an element counts only when its end tag follows within this many
#: characters of its start tag.
ELEMENT_TEXT_LIMIT = 512
#: A label takes at most this many further ;-separated lists, or lines below it, of tags.
TAG_LISTS_LIMIT = 64
#: Charsets that browsers decode as gb18030, the labels of GBK in the WHATWG Encoding
#: standard: a page declared as gb2312 often holds characters only GBK defines.
GB18030_CHARSETS = frozenset(
    {
        "chinese",
        "csgb2312",
        "csiso58gb231280",
        "gb2312",
        "gb_2312",
        "gb_2312-80",
        "gbk",
        "iso-ir-58",
        "x-gbk",
    }
)
PRIVATE_BANNER = "私有素材"
_RESTRICTIONS_KEY = "restrictions"
_IDENTITY_OVERRIDES = re.compile(r".*identity[-_]override.*", re.DOTALL)
_MANUAL_LABELS = re.compile(r"manual[_-][^/]*\.(?:json|jsonl|ndjson|ldjson)", re.IGNORECASE)
_JSON_SCHEMA_SUFFIX = ".schema.json"
# Content is read by the suffix of the file, or by the one before a backup suffix.
_JSON_DOCUMENT_SUFFIXES = frozenset({".json", ".geojson"})
_JSON_WITH_COMMENTS_SUFFIXES = frozenset({".json5", ".jsonc"})
_JSON_LINES_SUFFIXES = frozenset({".jsonl", ".ndjson", ".ldjson"})
_HTML_SUFFIXES = frozenset({".html", ".htm", ".shtml", ".xhtml", ".xht", ".svg", ".xml"})
_SAVED_PAGE_SUFFIXES = frozenset({".mht", ".mhtml"})
_CONTENT_SUFFIXES = (
    _JSON_DOCUMENT_SUFFIXES
    | _JSON_WITH_COMMENTS_SUFFIXES
    | _JSON_LINES_SUFFIXES
    | _HTML_SUFFIXES
    | _SAVED_PAGE_SUFFIXES
)
_BACKUP_SUFFIXES = frozenset({".bak", ".orig", ".old", ".backup", ".tmp", ".save"})
# Restriction statements in the text of a page: a list after a restrictions key (JSON,
# Python, YAML or TOML), and a string after a restrictions attribute or a quoted
# restrictions key (JSON in a page, a .jsonc or .json5 file with comments). The string
# ends at the quote that closes it, of the kind that opened it, and a backslash escapes
# the next character, as in JSON, so "partner's nda" is read whole. After = the string is
# also read without escapes, as HTML reads an attribute, so a value that ends in a
# backslash (data-restrictions="nda\") still closes at its quote. A list or string counts
# only when it closes within 512 characters, a string only when it holds no < or >. The
# patterns are tried only on pages that contain one of the label words.
_HTML_STATEMENTS = (
    re.compile(
        r"""restrictions?(?:[-_ ]?tags?)?["']?\s*[:=]\s*\[(?P<value>[^\]]{0,512})\]""",
        re.IGNORECASE,
    ),
    re.compile(
        r"""restrictions?(?:[-_ ]?tags?)?(?:\s*=|["']\s*[:=])\s*"""
        r"""(?P<quote>["'])(?P<value>(?:\\.|(?!(?P=quote))[^\\<>]){0,512})(?P=quote)""",
        re.IGNORECASE,
    ),
    re.compile(
        r"""restrictions?(?:[-_ ]?tags?)?\s*=\s*"""
        r"""(?P<quote>["'])(?P<value>(?:(?!(?P=quote))[^<>]){0,512})(?P=quote)""",
        re.IGNORECASE,
    ),
)
# A label and its value up to the end of its line, element or sentence: "Restrictions:"
# or "Restriction tags:", and the 限制标签 list of gbf's own reports. A value never
# starts on a later line, so a heading "Restrictions:" does not take the paragraph below
# it, only lines that hold nothing but tags (see _tag_lines_below).
_LABELS = (
    re.compile(
        r"\brestrictions?(?:[-_ ]?tags?)?[^\S\n]*[:：][^\S\n]*([^<>\n.;。；]{0,512})",
        re.IGNORECASE,
    ),
    re.compile(r"限制标签[^\S\n]*[：:][^\S\n]*([^）)<>\n]{0,512})"),
)
# A further ;-separated part of a label's value ("Restrictions: hl3; no_share").
_LABEL_SEGMENT = re.compile(r"[;；][^\S\n]*([^<>\n.;。；]{0,512})")
# A label that fills its line of the visible text.
_LABEL_LINE = re.compile(
    r"^[^\S\n]*(?:restrictions?(?:[-_ ]?tags?)?|限制标签)[^\S\n]*[:：][^\S\n]*$",
    re.IGNORECASE | re.MULTILINE,
)
_TAG_TOKEN = re.compile(r"[A-Za-z0-9_]+")
# What may separate the tags of a list: white space, commas, semicolons, slashes, bars,
# bullets, dashes, quotes and brackets.
_TAG_SEPARATORS = re.compile(
    r"[\s,，、;；/|+&*\-\N{MIDDLE DOT}\N{BULLET}\N{EN DASH}\N{EM DASH}\"'`()\[\]（）]*"
)
_TAG_LIST_LIMIT = 4096
_MARKUP = re.compile(r"<[^<>]*+>")
_SOURCE_LINE_BREAKS = str.maketrans("\r\n\t\f\v", "     ")
# Markup that ends a line of text; other markup (b, span, code, a, SVG text) does not.
_BLOCK_MARKUP = re.compile(
    r"</?(?:address|article|aside|blockquote|body|br|caption|dd|div|dl|dt|figcaption"
    r"|figure|footer|form|h[1-6]|head|header|hr|html|li|main|nav|ol|p|pre|section|table"
    r"|tbody|td|tfoot|th|thead|title|tr|ul)\b[^<>]*>",
    re.IGNORECASE,
)
# Markup a browser draws as nothing at all, not even a space.
_INVISIBLE_MARKUP = re.compile(r"</?wbr\b[^<>]*+>", re.IGNORECASE)
# Characters a browser does not draw: the soft hyphen (&shy;), zero-width spaces and
# joiners, direction marks, the word joiner, invisible operators, the byte order mark and
# tag characters.
_INVISIBLE = re.compile(
    r"[\N{SOFT HYPHEN}\N{COMBINING GRAPHEME JOINER}\N{ARABIC LETTER MARK}"
    r"\N{MONGOLIAN VOWEL SEPARATOR}\N{ZERO WIDTH SPACE}-\N{RIGHT-TO-LEFT MARK}"
    r"\N{LEFT-TO-RIGHT EMBEDDING}-\N{RIGHT-TO-LEFT OVERRIDE}\N{WORD JOINER}-\N{INVISIBLE PLUS}"
    r"\N{LEFT-TO-RIGHT ISOLATE}-\N{NOMINAL DIGIT SHAPES}\N{ZERO WIDTH NO-BREAK SPACE}"
    r"\N{LANGUAGE TAG}-\N{CANCEL TAG}]"
)
_WHITESPACE = re.compile(r"\s+")
_START_TAG = re.compile(r"<([A-Za-z][^\s/<>]*+)[^<>]*+>")
_END_TAG = re.compile(r"</([A-Za-z][^\s/<>]*+)\s*+>")
# An attribute name starts after a delimiter (the lookbehind) and is matched
# possessively. Without both, a long name that no "=" follows was rescanned from each of
# its characters, and a page of such tags took time quadratic in its length.
_ATTRIBUTE = re.compile(
    r"""(?<![^\s"'<>/=])([^\s"'<>/=]++)\s*+=\s*+(?:"([^"]*+)"|'([^']*+)'|([^\s"'<>]++))"""
)
# An element named restrictions (also with a prefix such as gbf:) states the tags in its
# content or value attribute, as <meta name="restrictions" content="..."> does, and in
# its text; an element whose class or id names restrictions, in its text.
_NAME_ATTRIBUTES = ("name", "property", "itemprop", "http-equiv")
_CLASS_ATTRIBUTES = ("class", "id")
_VALUE_ATTRIBUTES = ("content", "value")
_RESTRICTIONS_NAME = re.compile(r"(?:\S*[-_:.])?restrictions?(?:[-_]?tags?)?", re.IGNORECASE)
# Table and definition-list markup; a cell that reads "restrictions" labels the others.
_TABLE_MARKUP = re.compile(r"<(/?)(table|dl|tr|th|td|dt|dd)\b[^<>]*>", re.IGNORECASE)
_RESTRICTIONS_LABEL = re.compile(
    r"(?:restrictions?(?:[-_ ]?tags?)?|限制标签)\s*[:：]?", re.IGNORECASE
)
_RESTRICTION_WORD = re.compile(r"restriction|限制标签", re.IGNORECASE)
# The letters re.IGNORECASE matches to those of "restriction" besides their ASCII cases,
# İ and ı to i and ſ to s, which str.lower() does not map so.
_FOLDED_LETTERS = (
    "\N{LATIN CAPITAL LETTER I WITH DOT ABOVE}"
    "\N{LATIN SMALL LETTER DOTLESS I}"
    "\N{LATIN SMALL LETTER LONG S}"
)
# SVG text elements; one that reads "Restrictions" labels the ones after it.
_SVG_TEXT = re.compile(r"<(/?)text\b[^<>]*+>", re.IGNORECASE)
_EVIDENCE_DIRS = frozenset({"evidence", "evidence_store", "evidence-store"})
_EVIDENCE_SUFFIXES = frozenset(
    {
        ".jsonl",
        ".ndjson",
        ".parquet",
        ".sqlite",
        ".sqlite3",
        ".db",
        ".duckdb",
        ".arrow",
        ".feather",
        ".npz",
        ".h5",
        ".hdf5",
        ".lmdb",
    }
)

#: One content finding: (rule, detail, line).
Finding = tuple[str, str, int | None]


def is_media_file(rel: str) -> bool:
    return suffix(rel) in MEDIA_SUFFIXES


def _media_registry(ws: Workspace) -> tuple[dict[str, list[RegistryEntry]], str | None]:
    """Entries of registries/media.toml by path, and an error rule if it is unusable."""
    if MEDIA_REGISTRY not in ws.file_set:
        return {}, None
    try:
        raw = ws.read(MEDIA_REGISTRY)
        data = tomllib.loads(raw.decode("utf-8")) if raw is not None else {}
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return {}, "media-registry-unreadable"
    by_path: dict[str, list[RegistryEntry]] = {}
    for entry in registry_entries(data):
        by_path.setdefault(entry.path, []).append(entry)
    return by_path, None


def check_media(root: Path, workspace: Workspace | None = None) -> CheckResult:
    """Fail media/data/model/archive files that are unregistered, changed or restricted."""
    ws = workspace if workspace is not None else Workspace(root)
    result = CheckResult("media")
    targets = [rel for rel in ws.files if is_media_file(rel)]
    if not targets:
        return result
    registered, registry_error = _media_registry(ws)
    for rel in targets:
        try:
            if ws.size(rel) is None:
                continue
        except OSError as error:
            result.add(rel, "unreadable", detail=type(error).__name__)
            continue
        result.scanned += 1
        if registry_error is not None:
            result.add(rel, registry_error)
            continue
        entries = registered.get(rel, [])
        if not entries:
            result.add(rel, "unregistered")
            continue
        if len(entries) > 1:
            result.add(rel, "registered-twice", detail=f"{len(entries)} entries")
        entry = entries[0]
        if entry.sha256 is None:
            result.add(rel, "sha256-missing")
        else:
            try:
                actual = ws.sha256(rel)
            except OSError as error:
                result.add(rel, "unreadable", detail=type(error).__name__)
                continue
            if actual != entry.sha256.lower():
                result.add(rel, "sha256-mismatch")
        for tag in BLOCKED_RESTRICTIONS:
            if any(tag in e.restrictions for e in entries):
                result.add(rel, "restricted-tag", detail=tag)
    return result


def _is_evidence_store(parts: list[str]) -> bool:
    name = parts[-1].lower()
    if suffix(name) not in _EVIDENCE_SUFFIXES:
        return False
    if any(part.lower() in _EVIDENCE_DIRS for part in parts[:-1]):
        return True
    return name.startswith("evidence")


def _directory_rules(parts: list[str]) -> set[str]:
    """Rules a file breaks by the directories it sits in (at any depth)."""
    rules = set()
    for directory in {part.lower() for part in parts[:-1]}:
        rule = _RESTRICTED_DIRS.get(directory)
        if rule is None and _MANUAL_LABELS_DIR.fullmatch(directory):
            rule = "under-labels"
        if rule is None and _PLUGIN_JOB_DIR.fullmatch(directory):
            rule = "under-plugin-job"
        if rule is not None:
            rules.add(rule)
    return rules


def _name_rules(parts: list[str]) -> list[str]:
    """Rules a file breaks by its name alone (restricted material)."""
    name = parts[-1].lower()
    if name.endswith(NAME_RULE_EXEMPT_SUFFIXES):
        return []
    rules = []
    if _IDENTITY_OVERRIDES.fullmatch(name):
        rules.append("identity-overrides")
    if _MANUAL_LABELS.fullmatch(name):
        rules.append("manual-labels")
    if name in RAW_PLUGIN_OUTPUTS or any(p.fullmatch(name) for p in RAW_PLUGIN_OUTPUT_PATTERNS):
        rules.append("raw-plugin-output")
    return rules


def _is_json_schema(ws: Workspace, rel: str) -> bool:
    """Whether ``rel`` is a JSON Schema, which the name rules skip: a .schema.json file
    under JSON_SCHEMA_DIR, or one whose top level holds a "$schema" string."""
    if not rel.lower().endswith(_JSON_SCHEMA_SUFFIX):
        return False
    if rel.startswith(JSON_SCHEMA_DIR):
        return True
    try:
        size = ws.size(rel)
        data = ws.read(rel) if size is not None and size <= TEXT_SCAN_LIMIT else None
    except OSError:
        return False
    document = _parse_json(data)[0] if data is not None else None
    return isinstance(document, dict) and isinstance(document.get("$schema"), str)


@cache
def _known_tags() -> frozenset[str]:
    """The restriction tags gbf defines (src/gbf/contracts/schemas/restrictions.schema.json)."""
    return frozenset(contracts.load_schema("restrictions")["$defs"]["tag"]["enum"])


def _is_tag_list(text: str) -> bool:
    """Whether ``text`` holds tags that gbf defines and nothing else but separators."""
    if len(text) > _TAG_LIST_LIMIT:
        return False
    tokens = _TAG_TOKEN.findall(text)
    known = _known_tags()
    return (
        bool(tokens)
        and all(token.lower() in known for token in tokens)
        and _TAG_SEPARATORS.fullmatch(_TAG_TOKEN.sub("", text)) is not None
    )


def _own_tags(record: dict[str, Any]) -> set[str]:
    """Blocked tags under a ``restrictions`` key of one JSON object, in any letter case:
    the items of a list, or the words of a string ("no_share, nda")."""
    tags: set[str] = set()
    for key, value in record.items():
        if key.lower() != _RESTRICTIONS_KEY:
            continue
        values = [value] if isinstance(value, str) else value if isinstance(value, list) else []
        strings = [item for item in values if isinstance(item, str)]
        tags |= {token.lower() for item in strings for token in _TAG_TOKEN.findall(item)}
    return tags & set(BLOCKED_RESTRICTIONS)


def _records(document: Any) -> list[dict[str, Any]]:
    """The objects of one parsed JSON document that the content rules read: the top-level
    object and its object-valued members (one level down), or the same for each of the
    first JSON_ARRAY_SNIFF_ELEMENTS elements of a top-level array."""
    tops = document[:JSON_ARRAY_SNIFF_ELEMENTS] if isinstance(document, list) else [document]
    records = []
    for top in tops:
        if isinstance(top, dict):
            records.append(top)
            records += [value for value in top.values() if isinstance(value, dict)]
    return records


def _holds_hand_labels(record: dict[str, Any]) -> bool:
    """Whether a JSON object has the shape of the hand labels gbf.label_eval reads: a
    ``moments`` list one of whose first JSON_ARRAY_SNIFF_ELEMENTS items has every key of
    HAND_LABEL_MOMENT_KEYS."""
    moments = record.get("moments")
    return isinstance(moments, list) and any(
        isinstance(moment, dict) and HAND_LABEL_MOMENT_KEYS <= moment.keys()
        for moment in moments[:JSON_ARRAY_SNIFF_ELEMENTS]
    )


def _document_findings(document: Any) -> list[Finding]:
    """The blocked tags of one parsed JSON document (``restricted-tag``) and whether it
    holds hand labels (``manual-labels-content``), both read in its _records."""
    records = _records(document)
    findings = _tag_findings(set().union(*(_own_tags(record) for record in records)))
    if any(_holds_hand_labels(record) for record in records):
        findings.append(("manual-labels-content", "", None))
    return findings


def _decode(data: bytes) -> str | None:
    """The text of a JSON or HTML file in UTF-8, UTF-16 or UTF-32 (told apart by a BOM or
    zero bytes, as the json module does), or None when the bytes do not decode."""
    try:
        return data.decode(json.detect_encoding(data), "surrogatepass")
    except UnicodeDecodeError:
        return None


def _parse_json(data: bytes) -> tuple[Any, str | None, int | None]:
    """(document, None, None), or (None, rule, line) when ``data`` cannot be scanned:
    ``undecodable`` bytes, or ``json-parse-error`` for text that does not parse and
    mentions restrictions. Text that does not parse and never mentions them gives
    (None, None, None): it carries no tags the gate could read."""
    text = _decode(data)
    if text is None:
        return None, "undecodable", None
    try:
        return json.loads(text), None, None
    except (ValueError, RecursionError) as error:
        if _RESTRICTIONS_KEY in text.lower():
            return None, "json-parse-error", getattr(error, "lineno", None)
        return None, None, None


def _tag_findings(tags: set[str]) -> list[Finding]:
    return [("restricted-tag", tag, None) for tag in sorted(tags)]


def _json_document_findings(data: bytes) -> list[Finding]:
    document, rule, line = _parse_json(data)
    if rule is not None:
        return [(rule, "", line)]
    return _document_findings(document)


def _json_with_comments_findings(data: bytes) -> list[Finding]:
    """Findings in a .json5 or .jsonc file: read as JSON when it parses, else like the
    text of a page (its restriction statements; comments and trailing commas are legal
    there, so a parse error is not a failure)."""
    text = _decode(data)
    if text is None:
        return [("undecodable", "", None)]
    try:
        document = json.loads(text)
    except (ValueError, RecursionError):
        return _tag_findings(_statement_tags(_INVISIBLE.sub("", text)))
    return _document_findings(document)


def _json_lines_findings(head: bytes, *, complete: bool) -> list[Finding]:
    """Findings in the first JSON_LINES_SNIFF_RECORDS records of a JSON lines file.

    ``complete`` says whether ``head`` holds the whole file. If not, its last line may be
    cut short and is dropped, and the file fails as too-large-to-scan unless the head
    still holds JSON_LINES_SNIFF_RECORDS records. Blank lines are not records. A record
    that cannot be scanned is reported once per rule, at the first such line."""
    lines = head.split(b"\n")
    if not complete:
        lines = lines[:-1]
    records = [(number, line) for number, line in enumerate(lines, start=1) if line.strip()]
    records = records[:JSON_LINES_SNIFF_RECORDS]
    findings: list[Finding] = []
    if not complete and len(records) < JSON_LINES_SNIFF_RECORDS:
        detail = (
            f"{len(records)} of {JSON_LINES_SNIFF_RECORDS} records "
            f"in the first {JSON_LINES_SNIFF_BYTES} bytes"
        )
        findings.append(("too-large-to-scan", detail, None))
    found: set[Finding] = set()
    unreadable: set[str] = set()
    for number, line in records:
        document, rule, _ = _parse_json(line)
        if rule is not None and rule not in unreadable:
            unreadable.add(rule)
            findings.append((rule, "", number))
        found.update(_document_findings(document))
    return findings + sorted(found, key=lambda finding: finding[:2])


def _unescape(text: str) -> str:
    """``text`` with entities decoded and the characters a browser does not draw dropped."""
    return _INVISIBLE.sub("", html.unescape(text))


def _visible_text(page: str) -> str:
    """The text of a page as a browser lays it out: line breaks in the source are spaces,
    block elements end a line, <wbr> is nothing, other markup is a space, entities are
    decoded and characters a browser does not draw are dropped."""
    flat = _BLOCK_MARKUP.sub("\n", page.translate(_SOURCE_LINE_BREAKS))
    return _unescape(_MARKUP.sub(" ", _INVISIBLE_MARKUP.sub("", flat)))


def _tag_lines_below(text: str, position: int) -> list[str]:
    """The lines after the line that ends at ``position`` that hold nothing but tags, up to
    the first other line that is not blank (at most TAG_LISTS_LIMIT lines)."""
    lines = []
    start = position + 1
    for _ in range(TAG_LISTS_LIMIT):
        if start > len(text):
            break
        end = text.find("\n", start)
        end = len(text) if end < 0 else end
        line = text[start:end]
        if line.strip():
            if not _is_tag_list(line):
                break
            lines.append(line)
        start = end + 1
    return lines


def _label_values(text: str, *, lines: bool) -> list[str]:
    """The values of the labels in ``text`` (see _LABELS), each with the further
    ;-separated lists of tags after it; with ``lines`` (``text`` is the visible text of a
    page), also the lines of tags below a label that fills its line."""
    values = []
    for label in _LABELS:
        for match in label.finditer(text):
            values.append(match.group(1))
            end = match.end()
            for _ in range(TAG_LISTS_LIMIT):
                segment = _LABEL_SEGMENT.match(text, end)
                if segment is None or not _is_tag_list(segment.group(1)):
                    break
                values.append(segment.group(1))
                end = segment.end()
    if lines:
        for match in _LABEL_LINE.finditer(text):
            values += _tag_lines_below(text, match.end())
    return values


def _element_text(page: str, tag: re.Match[str]) -> str:
    """The visible text of the element that ``tag`` starts when its end tag follows within
    ELEMENT_TEXT_LIMIT characters; empty for a void, self-closing or unclosed element."""
    if tag.group().endswith("/>"):
        return ""
    name = tag.group(1).lower()
    window = page[tag.end() : tag.end() + ELEMENT_TEXT_LIMIT]
    for end in _END_TAG.finditer(window):
        if end.group(1).lower() == name:
            return _visible_text(window[: end.start()])
    return ""


def _named_values(page: str) -> list[str]:
    """The statements of elements named restrictions: the content and value attributes of
    one named so by its tag name or a name attribute, and the text of one named so by
    those or by its class or id, when that text holds nothing but tags."""
    values = []
    for tag in _START_TAG.finditer(page):
        if "restriction" not in tag.group().lower() and "&" not in tag.group():
            continue  # neither the name nor an entity that could spell it
        attributes = {}
        for match in _ATTRIBUTE.finditer(tag.group()):
            value = next(group for group in match.groups()[1:] if group is not None)
            attributes[match.group(1).lower()] = _unescape(value)
        names = [tag.group(1), *(attributes.get(name, "").strip() for name in _NAME_ATTRIBUTES)]
        named = any(_RESTRICTIONS_NAME.fullmatch(name) for name in names)
        if named:
            values += [attributes.get(name, "") for name in _VALUE_ATTRIBUTES]
        classes = [word for name in _CLASS_ATTRIBUTES for word in attributes.get(name, "").split()]
        if named or any(_RESTRICTIONS_NAME.fullmatch(name) for name in classes):
            text = _element_text(page, tag)
            if _is_tag_list(text):
                values.append(text)
    return values


def _table_values(page: str) -> list[str]:
    """The cell texts of every table or definition list one of whose cells reads
    "restrictions" (or "restriction tags", 限制标签): that cell is a label, so the others
    may state the tags. Cells outside a table count as one table; a table that is never
    closed ends the page."""
    tables: list[list[tuple[int, int]]] = []  # cell spans of the open tables, innermost last
    closed: list[list[tuple[int, int]]] = []
    marks = list(_TABLE_MARKUP.finditer(page))
    for index, mark in enumerate(marks):
        closing, element = mark.group(1), mark.group(2).lower()
        if element in ("table", "dl"):
            if not closing:
                tables.append([])
            elif tables:
                closed.append(tables.pop())
            continue
        if closing or element == "tr":
            continue
        if not tables:
            tables.append([])
        end = marks[index + 1].start() if index + 1 < len(marks) else len(page)
        tables[-1].append((mark.end(), end))

    def is_label(start: int, end: int) -> bool:
        if not (
            _RESTRICTION_WORD.search(page, start, end)
            or page.find("&", start, end) >= 0
            or _INVISIBLE_MARKUP.search(page, start, end)
        ):
            return False  # neither the word nor an entity or <wbr> that could spell it
        return bool(_RESTRICTIONS_LABEL.fullmatch(_visible_text(page[start:end]).strip()))

    return [
        " ".join(_visible_text(page[start:end]) for start, end in spans)
        for spans in closed + tables
        if any(is_label(start, end) for start, end in spans)
    ]


def _svg_values(page: str) -> list[str]:
    """The SVG text elements that hold nothing but tags and follow one that reads
    "Restrictions" (or "Restriction tags", 限制标签, with or without a colon)."""
    marks = list(_SVG_TEXT.finditer(page))
    values = []
    labelled = False
    for index, mark in enumerate(marks):
        if mark.group(1):
            continue  # an end tag
        end = marks[index + 1].start() if index + 1 < len(marks) else len(page)
        content = page[mark.end() : end]
        if not labelled and not (
            _RESTRICTION_WORD.search(content) or "&" in content or _INVISIBLE_MARKUP.search(content)
        ):
            continue  # not a label, and no label before it
        text = _visible_text(content).strip()
        if _RESTRICTIONS_LABEL.fullmatch(text):
            labelled = True
        elif labelled and _is_tag_list(text):
            values.append(text)
        else:
            labelled = False
    return values


def _has_label_word(text: str, lowered: str) -> bool:
    """Whether _RESTRICTION_WORD occurs in ``text`` (``lowered`` is ``text.lower()``).
    Substring searches answer that many times faster than the expression on a large
    page; a text that holds one of _FOLDED_LETTERS is searched with the expression."""
    if "限制标签" in text or "restriction" in lowered:
        return True
    if any(letter in text for letter in _FOLDED_LETTERS):
        return _RESTRICTION_WORD.search(text) is not None
    return False


def _statement_tags(page: str) -> set[str]:
    """Blocked tags that the restriction statements of a page state (see the module
    docstring); ``page`` holds no characters that a browser does not draw."""
    text = _unescape(page)
    lowered = text.lower()
    labelled = _has_label_word(text, lowered)
    # Other markup is a space in the visible text, so only <wbr>, which is nothing there,
    # can join a label word that the source does not hold (Re<wbr>strictions); the text
    # keeps the markup of the page, so it shows every <wbr>.
    if not labelled and "<wbr" not in lowered and "</wbr" not in lowered:
        return set()
    visible = _visible_text(page)
    if not (labelled or _RESTRICTION_WORD.search(visible)):
        return set()  # no label word, in the source or as drawn
    statements = [
        match.group("value")
        for source in (text, visible)
        for pattern in _HTML_STATEMENTS
        for match in pattern.finditer(source)
    ]
    statements += _label_values(text, lines=False) + _label_values(visible, lines=True)
    statements += _named_values(page) + _table_values(page) + _svg_values(page)
    tokens = {token.lower() for statement in statements for token in _TAG_TOKEN.findall(statement)}
    return tokens & set(BLOCKED_RESTRICTIONS)


def _page_findings(page: str) -> list[Finding]:
    """Blocked tags a page states, and its private-material banner (see the module
    docstring)."""
    page = _INVISIBLE.sub("", page)
    findings = _tag_findings(_statement_tags(page))
    text = _unescape(page)
    if PRIVATE_BANNER in text or (
        all(character in text for character in PRIVATE_BANNER)
        and PRIVATE_BANNER in _WHITESPACE.sub("", _unescape(_MARKUP.sub("", page)))
    ):
        findings.append(("private-banner", "", None))
    return findings


def _html_findings(data: bytes) -> list[Finding]:
    page = _decode(data)
    if page is None:
        return [("undecodable", "", None)]
    return _page_findings(page)


def _saved_page_findings(data: bytes) -> list[Finding]:
    """Findings in a saved page (.mht, .mhtml, a MIME message), read like a page: the file
    as it is when it decodes (its headers, parts in plain text), and each text part decoded
    as its headers declare (base64 or quoted-printable, charset; GB18030_CHARSETS as
    gb18030), a part that declares no transfer encoding also as quoted-printable. Parts
    nested too deeply for the email parser, unusable header parameters (including
    malformed RFC 2231 continuations), and a part whose charset cannot be used at all
    fail closed (``mime-parse-error``). The raw page is still checked if MIME parsing
    fails; a part with an unusable charset is still read as UTF-8, UTF-16 or UTF-32."""
    raw = _decode(data)
    findings: set[Finding] = set(_page_findings(raw)) if raw is not None else set()
    try:
        parts = list(email.message_from_bytes(data).walk())
    except RecursionError:
        findings.add(("mime-parse-error", "parts nested too deeply", None))
        parts = []
    except (ValueError, TypeError):
        findings.add(("mime-parse-error", "unusable header parameter", None))
        parts = []
    for part in parts:
        if part.is_multipart() or part.get_content_maintype() != "text":
            continue
        payload = part.get_payload(decode=True)
        if not isinstance(payload, bytes):
            continue
        bodies = [payload]
        if part.get("content-transfer-encoding") is None:
            bodies.append(quopri.decodestring(payload))
        try:
            charset = part.get_content_charset()
        except (ValueError, TypeError):  # unusable RFC 2231 charset or continuations
            findings.add(("mime-parse-error", "unusable charset", None))
            charset = None
        if charset in GB18030_CHARSETS:
            charset = "gb18030"
        for body in bodies:
            page = None
            if charset is not None:
                try:
                    page = body.decode(charset)
                except (LookupError, UnicodeDecodeError):
                    page = None
                except ValueError:  # a charset name with a NUL, a codec that fails otherwise
                    findings.add(("mime-parse-error", "unusable charset", None))
            page = page if page is not None else _decode(body)
            if page is None:
                findings.add(("undecodable", "", None))
            else:
                findings.update(_page_findings(page))
    return sorted(findings, key=lambda finding: finding[:2])


def _content_kind(rel: str) -> str:
    """The suffix that says how to read a file: its last one, or the one before a backup
    suffix (report.html.bak and report.html~ are read as .html)."""
    kinds = suffixes(rel.removesuffix("~"))
    while len(kinds) > 1 and kinds[-1] in _BACKUP_SUFFIXES:
        kinds.pop()
    return kinds[-1] if kinds else ""


def _check_content(ws: Workspace, rel: str, result: CheckResult) -> None:
    """Add the problems found in the content of a JSON, JSON lines, HTML, SVG, XML or saved
    page file."""
    kind = _content_kind(rel)
    if kind not in _CONTENT_SUFFIXES:
        return
    try:
        size = ws.size(rel)
        if size is None:
            return
        if kind in _JSON_LINES_SUFFIXES:
            data = ws.head(rel, JSON_LINES_SNIFF_BYTES)
        elif size > TEXT_SCAN_LIMIT:
            result.add(rel, "too-large-to-scan", detail=f"{size} bytes")
            return
        else:
            data = ws.read(rel)
    except OSError as error:
        result.add(rel, "unreadable", detail=type(error).__name__)
        return
    if data is None:
        return
    if kind in _JSON_LINES_SUFFIXES:
        findings = _json_lines_findings(data, complete=size <= len(data))
    elif kind in _JSON_DOCUMENT_SUFFIXES:
        findings = _json_document_findings(data)
    elif kind in _JSON_WITH_COMMENTS_SUFFIXES:
        findings = _json_with_comments_findings(data)
    elif kind in _SAVED_PAGE_SUFFIXES:
        findings = _saved_page_findings(data)
    else:
        findings = _html_findings(data)
    for rule, detail, line in findings:
        result.add(rel, rule, line=line, detail=detail)


def check_restricted_data(root: Path, workspace: Workspace | None = None) -> CheckResult:
    """Fail restricted data by path, name or content (see the module docstring)."""
    ws = workspace if workspace is not None else Workspace(root)
    result = CheckResult("restricted_data")
    for rel in ws.files:
        parts = rel.split("/")
        result.scanned += 1
        for rule in sorted(_directory_rules(parts)):
            result.add(rel, rule)
        name_rules = _name_rules(parts)
        if name_rules and not _is_json_schema(ws, rel):
            for rule in name_rules:
                result.add(rel, rule)
        _check_content(ws, rel, result)
        if _STATSBOMB_PATH.search(rel):
            result.add(rel, "statsbomb-path")
        elif suffix(rel) in _JSON_SUFFIXES:
            try:
                head = ws.head(rel, STATSBOMB_SNIFF_BYTES)
            except OSError as error:
                result.add(rel, "unreadable", detail=type(error).__name__)
                head = None
            if head is not None and all(marker in head for marker in _STATSBOMB_MARKERS):
                result.add(rel, "statsbomb-content")
        if _SOCCERNET_LABELS.fullmatch(basename(rel)):
            result.add(rel, "soccernet-labels")
        if any(s in PICKLE_SUFFIXES for s in suffixes(rel)):
            result.add(rel, "pickle-file")
        if _is_evidence_store(parts):
            result.add(rel, "evidence-store")
    return result
