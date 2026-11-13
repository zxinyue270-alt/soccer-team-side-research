# Gate configuration

`python -m gbf gate` runs the offline static checks; `python -m gbf gate --full` adds ruff
and the test suite (plan section 7, G1-G4, G16, G17). The exit code is 0 only when every
check passes. Problems print as `<check> <relative path>[:line] <rule id>`; the gate never
prints the matched text of a secret or a local path. `--json PATH` writes the run as a
receipt of kind `gate`.

Checks run over the candidate files: tracked files plus untracked files that are not
ignored. Static checks: `eol`, `credentials`, `abs_paths`, `media`, `restricted_data`,
`forbidden_imports`, `deps`, `registries`, `receipts`. The rules are documented in the
module docstrings under `src/gbf/gate/`.

## Files in this directory

- `registries.json`: registry kinds that must exist as `registries/<kind>.toml`. Each
  registry must also validate against schema `urn:gbf:schema:registry_<kind>`.
- `min_counts.json`: minimum number of tests per suite, where a suite is a test file path
  relative to `tests/`. `--full` fails when a suite has fewer tests than its minimum, and
  prints a hint when a minimum can be raised or a suite has no minimum yet.

## Lowering a minimum

Minimums only go up. `--full` compares `min_counts.json` with the version at
`--counts-base` (default `HEAD`; CI should pass the target branch). Lowering a minimum, or
removing a suite, fails unless `exemptions` holds an entry for exactly that change:

```json
{"suite": "test_example.py", "from": 12, "to": 10,
 "reason": "two cases merged into one parametrised test",
 "adr": "docs/adr/0005-example.md"}
```

The ADR file must exist in the repository.

## The test run

Tests run in a subprocess with a sanitised environment: only what Python, git and ffmpeg
need (PATH, SYSTEMROOT, TEMP, HOME, LANG, ...) plus `GBF_*` variables survive, and
`PYTHONUTF8=1` and `PYTHONHASHSEED=0` are forced. A generated `sitecustomize` blocks
network access in every Python process of the run, including processes that tests start.
Any failure, error, skip, xfail, xpass or deselected test fails the gate.
