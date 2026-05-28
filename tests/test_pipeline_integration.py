"""
Slow end-to-end pipeline test. Hits the live BacDive API for 10 IDs.

Skipped by default — run explicitly with:
    python -m pytest tests/test_pipeline_integration.py -v
    python -m pytest -m integration -v

What it verifies:
    fetch -> parse -> splits -> vocab -> model.prepare_labels -> single forward+backward
    All in one pytest run, using a temporary data directory.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    """Redirect DATA_DIR in the scripts to a temp directory for isolation."""
    d = tmp_path / "data"
    d.mkdir()
    # The scripts read DATA paths at module import; we monkeypatch by symlinking
    # tmp data over the real one is too invasive. Instead, run scripts as subprocesses
    # with a working directory of tmp_path and a symlink to the project for code.
    return d


pytestmark = pytest.mark.integration


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Run a shell command and capture output, failing the test on non-zero exit."""
    result = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        print("STDOUT:\n" + result.stdout, file=sys.stderr)
        print("STDERR:\n" + result.stderr, file=sys.stderr)
        pytest.fail(f"command failed: {' '.join(cmd)}")
    return result


def test_end_to_end_smoke(tmp_path):
    """
    Fetch 10 BacDive IDs, parse them, build splits + vocab, do prepare_labels,
    do one model forward+backward. No GPU, no ESM-2, no genome downloads.
    """
    # Stage a temp working directory with all repo files but a fresh data/ dir
    work = tmp_path / "repo"
    work.mkdir()
    for item in ROOT.iterdir():
        if item.name in (".git", "data", "runs", ".pytest_cache", "__pycache__"):
            continue
        target = work / item.name
        if item.is_dir():
            target.symlink_to(item.resolve())
        else:
            target.symlink_to(item.resolve())
    (work / "data").mkdir()

    # 1. Fetch a small slice. Need enough strains that family-split produces
    # non-empty val + test buckets — 10 is too few; 100 reliably works.
    run([sys.executable, "fetch_bacdive.py", "--start", "1", "--end", "100",
         "--workers", "5", "--batch", "50"], cwd=work)
    raw = (work / "data" / "bacdive_raw.jsonl")
    assert raw.exists() and raw.stat().st_size > 0, "fetch produced no records"

    # 2. Parse
    run([sys.executable, "parse_bacdive.py"], cwd=work)
    traits = work / "data" / "traits.parquet"
    assert traits.exists()

    # 3. Splits
    run([sys.executable, "splits.py"], cwd=work)
    splits = work / "data" / "splits.parquet"
    assert splits.exists()

    # 4. Vocab
    run([sys.executable, "vocab.py"], cwd=work)
    vocab = work / "data" / "vocabularies.json"
    assert vocab.exists()
    v = json.loads(vocab.read_text())
    assert "vocabularies" in v
    assert "gram_stain" in v["vocabularies"]

    # 5. Model smoke — 1 epoch with random features
    run([sys.executable, "model.py", "--epochs", "1", "--feat-dim", "16",
         "--batch", "4", "--save-metrics", "runs/integration.json",
         "--run-name", "integration"], cwd=work)
    metrics_path = work / "runs" / "integration.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text())
    assert metrics["run_name"] == "integration"
    assert metrics["n_test"] >= 0
    assert "per_head" in metrics
    # At least one head should have a numeric score
    assert any(isinstance(h["score"], (int, float)) for h in metrics["per_head"].values())
