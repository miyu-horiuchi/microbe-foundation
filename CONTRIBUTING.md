# Contributing to microbe-foundation

Thanks for your interest. This document covers **code contributions** (PRs that change the codebase). For **benchmark submissions** (adding your model's scores to the leaderboard), see `BENCHMARK.md` instead.

## Quick start

```bash
git clone https://github.com/miyu-horiuchi/microbe-foundation
cd microbe-foundation
pip install -r requirements.txt
pip install pytest
python -m pytest tests/ -v        # 45 tests, ~2 s; should all pass
```

## How to propose a change

1. **Open an issue first** for anything non-trivial (new trait head, schema change, encoder addition, evaluation-protocol tweak). Lets us discuss scope before you invest in code.
2. **Fork + branch** with a descriptive name (e.g. `add-polar-lipid-head`, `fix-halophily-binning`).
3. **Run the test suite** (`python -m pytest tests/ -v`) and make sure it still passes. If your change adds or modifies behavior covered by a test, update the test in the same PR.
4. **Run the schema sanity check** (`python schema.py`) if you touched `schema.py` — it must regenerate `trait_schema.json` cleanly.
5. **Open a PR** against `main` with a description that names the user-visible change. CI runs the full pytest suite on Python 3.11 and 3.12.

## What kinds of changes are welcome

- **New trait heads** (e.g., polar lipids once an IJSEM PDF-mining script exists). Touch `schema.py`, add an extractor in `parse_bacdive.py`, register in `model.py`'s prepare_labels via the schema, add tests in `tests/test_parse.py`.
- **New feature encoders** (KO baseline, Bacformer extension, Evo 2 wrapper). Add a `compute_<encoder>_features.py` that produces the standard `.npz` shape (`bacdive_ids` int64, `features` float32).
- **Parser robustness** — if you find a BacDive field shape that breaks an extractor, add a fixture record to `tests/conftest.py` and a regression test.
- **Better baselines** — drop in a new model architecture in `model.py` (or a sibling file) keeping the schema-driven head construction.
- **Documentation** — typos, clearer examples, missing context. Always welcome.

## What needs more discussion before a PR

- **Schema changes** (renaming/removing heads, changing head types). These break the benchmark protocol — bump the schema version and bring it up in an issue first.
- **Split changes** (different seed, different fractions, different group key). Affects every reported score.
- **Vocabulary changes** (different top-N for medium / FAME / etc.). Breaks comparability across submissions.

## Code style

- **No emojis** in code or doc unless explicitly requested.
- **No comments explaining what the code does** — well-named identifiers do that. Comments are for *why*: hidden constraints, surprising behavior, references to specific BacDive field quirks.
- **No new docs / READMEs** unless adding a genuinely new artifact. Edit existing files when possible.
- **Type hints** are encouraged (`from __future__ import annotations` at the top so `dict | None` etc. works on Python 3.9). Don't go overboard with generics.
- **No emoji in commit messages**. Use the prefix conventions visible in `git log`: `Phase N:`, `module: short description`.

## Running the full pipeline locally

```bash
bash scripts/run_all.sh --smoke    # ~5 minutes end-to-end on 1k records
```

The smoke run hits IDs 1–1000, uses the smallest ESM-2 (8M params), and trains 3 epochs. Useful for verifying a code change end-to-end before pushing.

## Architecture in one paragraph

`schema.py` is the source of truth for what gets predicted. `parse_bacdive.py` is the source of truth for how BacDive's JSON maps to those traits. `splits.py` and `vocab.py` are deterministic given the parsed parquet. `model.py` builds heads dynamically from `trait_schema.json` + `vocabularies.json` — so adding a head is a 3-line change in schema.py + a new extractor in parse_bacdive.py, and the model + paper tables update automatically.

## Questions

Open an issue with the `question` label. There's no separate forum.

## License

By contributing you agree your work is licensed under the same MIT license that covers the rest of the codebase.
