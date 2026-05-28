"""Streaming fetch+featurize pipeline.

Each worker:
  1. Downloads a genome FASTA from NCBI Datasets into memory (no disk write)
  2. Runs pyrodigal + AA-composition feature extraction
  3. Returns the feature dict — caller persists to a JSONL append log

Workers are fully independent processes; the only shared state is the JSONL log
(written from the parent), so resumability is trivial: skip any bacdive_id whose
row is already in the log.
"""
from __future__ import annotations

import io
import json
import time
import zipfile
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

from microbe_model import config
from microbe_model.features.genome import extract_features_from_seqs

DATASETS_URL = "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/{acc}/download"
RATE_LIMIT_S = 0.1 if config.NCBI_API_KEY else 0.34
VERSION_FALLBACKS = (".1", ".2", ".3", ".4")  # tried in order for unversioned accessions
EMPTY_ZIP_BYTES = 2_000  # NCBI "no content" zips are ~850 bytes; real genomes are MB+


def _has_version(accession: str) -> bool:
    """True if the accession ends in `.<digit(s)>` (e.g. GCA_X.1, GCF_X.2)."""
    if "." not in accession:
        return False
    suffix = accession.rsplit(".", 1)[-1]
    return suffix.isdigit()


def _candidate_accessions(accession: str) -> list[str]:
    if _has_version(accession):
        return [accession]
    return [accession + v for v in VERSION_FALLBACKS]


def _fetch_one_accession(accession: str) -> bytes | None:
    """Download a single (versioned) accession and return zip bytes, or None on miss."""
    headers: dict[str, str] = {"Accept": "application/zip"}
    if config.NCBI_API_KEY:
        headers["api-key"] = config.NCBI_API_KEY
    params = {"include_annotation_type": "GENOME_FASTA"}

    for attempt in range(3):
        try:
            time.sleep(RATE_LIMIT_S)
            resp = requests.get(
                DATASETS_URL.format(acc=accession),
                params=params,
                headers=headers,
                timeout=120,
            )
            if resp.status_code == 404:
                return None
            if resp.status_code in (429, 502, 503):
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
        except requests.RequestException:
            if attempt == 2:
                return None
            time.sleep(2 ** attempt)
            continue

        # NCBI returns 200 + tiny "empty" zip when the accession doesn't exist
        # (e.g. unversioned forms or version that was never assigned).
        if len(resp.content) < EMPTY_ZIP_BYTES:
            return None
        return resp.content
    return None


def _fetch_fasta_bytes(accession: str) -> list[tuple[str, str]] | None:
    """Download a genome FASTA and return [(contig_id, sequence_str), ...].

    For unversioned accessions, tries ``.1``, ``.2``, ``.3``, ``.4`` in order
    (BacDive stores accessions without version suffixes; we resolve to the actual
    deposited version). Returns None if no version yields data.
    """
    for candidate in _candidate_accessions(accession):
        zip_bytes = _fetch_one_accession(candidate)
        if zip_bytes is None:
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                fasta_names = [n for n in zf.namelist() if n.endswith(".fna")]
                if not fasta_names:
                    continue
                with zf.open(fasta_names[0]) as src:
                    raw = src.read()
        except zipfile.BadZipFile:
            continue
        return _parse_fasta_bytes(raw)
    return None


def _parse_fasta_bytes(raw: bytes) -> list[tuple[str, str]]:
    """Minimal in-memory FASTA parser, returns list of (id, sequence) tuples.

    Avoids biopython's overhead and the gzip→tempfile→biopython round-trip.
    """
    contigs: list[tuple[str, str]] = []
    current_id: str | None = None
    current_chunks: list[str] = []
    for line_bytes in raw.splitlines():
        if not line_bytes:
            continue
        if line_bytes.startswith(b">"):
            if current_id is not None:
                contigs.append((current_id, "".join(current_chunks).upper()))
            current_id = line_bytes[1:].decode("ascii", errors="replace").split()[0]
            current_chunks = []
        else:
            current_chunks.append(line_bytes.decode("ascii", errors="replace"))
    if current_id is not None:
        contigs.append((current_id, "".join(current_chunks).upper()))
    return contigs


def _process_one(args: tuple[int, str]) -> dict[str, Any] | None:
    """Worker entry point — runs in a child process. Returns None on any failure."""
    bacdive_id, accession = args
    contigs = _fetch_fasta_bytes(accession)
    if not contigs:
        return None
    try:
        feats = extract_features_from_seqs(contigs)
    except Exception:
        return None
    feats["bacdive_id"] = bacdive_id
    feats["genome_accession"] = accession
    return feats


def stream_fetch_and_featurize(
    tasks: list[tuple[int, str]],
    *,
    out_path: Path,
    n_workers: int,
    on_progress: Callable[[int, int, int], None] | None = None,
) -> None:
    """Fetch + featurize a list of (bacdive_id, accession) pairs in parallel.

    Streams successful results as JSON lines into out_path. Skips tasks already in the file.
    """
    done_ids = _load_done_ids(out_path)
    pending = [(bid, acc) for bid, acc in tasks if bid not in done_ids]

    if not pending:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as log, ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_process_one, t): t for t in pending}
        n_success = 0
        for n_completed, future in enumerate(as_completed(futures), start=1):
            try:
                result = future.result()
            except Exception:
                result = None
            if result is not None:
                log.write(json.dumps(result) + "\n")
                log.flush()
                n_success += 1
            if on_progress is not None:
                on_progress(n_completed, n_success, len(pending))


def _load_done_ids(jsonl_path: Path) -> set[int]:
    if not jsonl_path.exists():
        return set()
    done: set[int] = set()
    with open(jsonl_path) as fh:
        for line in fh:
            try:
                row = json.loads(line)
                done.add(int(row["bacdive_id"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return done


