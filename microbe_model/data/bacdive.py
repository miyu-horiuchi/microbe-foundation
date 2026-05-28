"""BacDive REST API client (v2, public).

The BacDive v2 API is fully open as of February 2026 — no registration, no auth.
Documentation: https://api.bacdive.dsmz.de/

We discover strain IDs by scanning the integer ID space in semicolon-batched fetches
of up to 100 IDs per call. Missing IDs are silently dropped server-side, so a blind
scan over [1, MAX_ID] yields every existing record in one pass. At ~150K live IDs
(as of 2026-04), this takes ~30 minutes single-threaded.
"""
from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import requests

from microbe_model import config

BASE_URL = "https://api.bacdive.dsmz.de/v2"
BATCH_SIZE = 100  # max IDs per /fetch/ call (server limit)
DEFAULT_MAX_ID = 200_000  # conservative upper bound; live max is ~160K-180K as of 2026-04


class BacDiveClient:
    def __init__(self, *, request_timeout: int = 60, retry_sleep_s: float = 1.0) -> None:
        self._session = requests.Session()
        self.timeout = request_timeout
        self.retry_sleep_s = retry_sleep_s

    def _get(self, path: str, params: dict | None = None) -> dict[str, Any]:
        url = f"{BASE_URL}{path}"
        for attempt in range(3):
            resp = self._session.get(url, params=params, timeout=self.timeout)
            if resp.status_code == 429:
                time.sleep(self.retry_sleep_s * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        resp.raise_for_status()
        return {}

    def fetch_batch(self, ids: list[int]) -> dict[int, dict[str, Any]]:
        """Fetch up to BATCH_SIZE strain records in a single call.

        Returns a {bacdive_id: record} mapping. Missing IDs are absent from the result.
        """
        if not ids:
            return {}
        if len(ids) > BATCH_SIZE:
            raise ValueError(f"Batch size {len(ids)} exceeds server limit {BATCH_SIZE}")
        path = f"/fetch/{';'.join(str(i) for i in ids)}"
        body = self._get(path)
        results = body.get("results")
        if isinstance(results, dict):
            return {int(k): v for k, v in results.items()}
        return {}

    def iter_records(
        self,
        *,
        start: int = 1,
        end: int = DEFAULT_MAX_ID,
        batch_size: int = BATCH_SIZE,
    ) -> Iterator[tuple[int, dict[str, Any]]]:
        """Scan the BacDive ID range and yield (id, record) for every existing strain."""
        for batch_start in range(start, end + 1, batch_size):
            batch_end = min(batch_start + batch_size - 1, end)
            ids = list(range(batch_start, batch_end + 1))
            records = self.fetch_batch(ids)
            yield from sorted(records.items())


def cache_path(bacdive_id: int) -> Path:
    return config.BACDIVE_DIR / f"{bacdive_id}.json"


def cache_record(bacdive_id: int, record: dict[str, Any]) -> Path:
    path = cache_path(bacdive_id)
    path.write_text(json.dumps(record))
    return path


def load_cached(bacdive_id: int) -> dict[str, Any] | None:
    path = cache_path(bacdive_id)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def extract_phenotypes(record: dict[str, Any]) -> dict[str, Any]:
    """Pull the v0 prediction targets out of a BacDive v2 record.

    Field locations (verified against live API on 2026-04-26):
      - General → BacDive-ID
      - Name and taxonomic classification → species, genus, family
      - Culture and growth conditions → culture temp[] (type ∈ {growth, optimum, range, no growth})
      - Culture and growth conditions → culture pH[] (same shape)
      - Physiology and metabolism → oxygen tolerance[]
      - Physiology and metabolism → halophily[]
      - Sequence information → Genome sequences[].INSDC accession
      - Isolation, sampling and environmental information → isolation source categories[].Cat{1,2,3}
    """
    general = record.get("General") or {}
    taxon = record.get("Name and taxonomic classification") or {}
    culture = record.get("Culture and growth conditions") or {}
    physio = record.get("Physiology and metabolism") or {}
    seq = record.get("Sequence information") or {}
    iso = record.get("Isolation, sampling and environmental information") or {}

    iso_cats = _collect_isolation_categories(iso.get("isolation source categories"))

    out: dict[str, Any] = {
        "bacdive_id": general.get("BacDive-ID"),
        "species": taxon.get("species"),
        "genus": taxon.get("genus"),
        "family": (taxon.get("LPSN") or {}).get("family") or taxon.get("family"),
        "ncbi_taxon_id": _first_ncbi_tax_id(general.get("NCBI tax id")),
        "optimal_temperature_c": _derive_optimum(_as_list(culture.get("culture temp")), "temperature"),
        "optimal_ph": _derive_optimum(_as_list(culture.get("culture pH")), "pH"),
        "oxygen_requirement": _first_value(_as_list(physio.get("oxygen tolerance")), "oxygen tolerance"),
        "salt_tolerance_pct": _derive_salt(physio.get("halophily")),
        "genome_accession": _first_genome_accession(seq.get("Genome sequences")),
        "isolation_cat1": iso_cats["cat1"],
        "isolation_cat2": iso_cats["cat2"],
        "isolation_cat3": iso_cats["cat3"],
    }
    return out


def _collect_isolation_categories(raw: Any) -> dict[str, str | None]:
    """Flatten BacDive's `isolation source categories` into 3 pipe-joined string fields.

    A strain commonly has multiple parallel category descriptions (e.g., #Host=Human AND
    #Host Body Product=Blood). We collect *all* unique values per level into a sorted,
    pipe-joined string so downstream code can split & one-hot. The leading '#' is stripped.
    """
    cats: dict[str, set[str]] = {"Cat1": set(), "Cat2": set(), "Cat3": set()}
    for entry in _as_list(raw):
        if not isinstance(entry, dict):
            continue
        for level in cats:
            value = entry.get(level)
            if isinstance(value, str) and value:
                cats[level].add(value.lstrip("#").strip())
    return {
        "cat1": "|".join(sorted(cats["Cat1"])) or None,
        "cat2": "|".join(sorted(cats["Cat2"])) or None,
        "cat3": "|".join(sorted(cats["Cat3"])) or None,
    }


def _as_list(x: Any) -> list:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _to_float(x: Any) -> float | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    if "-" in s and not s.startswith("-"):
        # e.g. "5-30" — return midpoint
        parts = s.split("-")
        try:
            lo, hi = float(parts[0]), float(parts[1])
            return (lo + hi) / 2
        except (ValueError, IndexError):
            return None
    try:
        return float(s.split()[0])
    except (ValueError, AttributeError):
        return None


def _derive_optimum(entries: list, value_key: str) -> float | None:
    """Find an optimum for a temperature- or pH-like list of {type, value} entries.

    Preference order:
      1. type == "optimum" (exact)
      2. median of "positive growth" entries
      3. None
    """
    optima = []
    growth = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        etype = (entry.get("type") or "").lower()
        value = _to_float(entry.get(value_key))
        if value is None:
            continue
        is_positive = (entry.get("growth") or "").lower() in {"positive", "yes", "+", "true"}
        if "optim" in etype:
            optima.append(value)
        elif etype == "growth" and is_positive:
            growth.append(value)
    if optima:
        return sum(optima) / len(optima)
    if growth:
        sorted_g = sorted(growth)
        n = len(sorted_g)
        return sorted_g[n // 2] if n % 2 else (sorted_g[n // 2 - 1] + sorted_g[n // 2]) / 2
    return None


def _first_value(entries: list, key: str) -> str | None:
    for entry in entries:
        if isinstance(entry, dict) and entry.get(key):
            return str(entry[key])
    return None


def _derive_salt(halophily: Any) -> float | None:
    """Derive optimal NaCl concentration (% w/v) from BacDive halophily entries.

    Each entry has shape:
      {salt: 'NaCl', growth: 'positive'|'no', tested relation: 'optimum'|'growth', concentration: '3 %'}

    Preference order (mirrors _derive_optimum):
      1. tested relation == 'optimum' AND growth == 'positive'
      2. median of positive-growth concentrations (the strain's tolerated range)
      3. None

    The previous implementation returned the first parsable value, which often picked
    the lowest tested concentration or a no-growth entry — overstating salt sensitivity.
    """
    positive_tokens = {"positive", "yes", "+", "true"}
    optima: list[float] = []
    growth: list[float] = []
    for entry in _as_list(halophily):
        if not isinstance(entry, dict):
            continue
        if (entry.get("salt") or "NaCl") != "NaCl":
            continue
        is_positive = (entry.get("growth") or "").lower() in positive_tokens
        relation = (entry.get("tested relation") or "").lower()
        value = _to_float(entry.get("concentration") or entry.get("salt concentration"))
        if value is None:
            continue
        if "optim" in relation and is_positive:
            optima.append(value)
        elif is_positive:
            growth.append(value)
    if optima:
        return sum(optima) / len(optima)
    if growth:
        sorted_g = sorted(growth)
        n = len(sorted_g)
        return sorted_g[n // 2] if n % 2 else (sorted_g[n // 2 - 1] + sorted_g[n // 2]) / 2
    return None


def _first_genome_accession(genome_entries: Any) -> str | None:
    for entry in _as_list(genome_entries):
        if isinstance(entry, dict):
            for key in ("INSDC accession", "NCBI accession", "accession"):
                value = entry.get(key)
                if value:
                    return str(value)
    return None


def _first_ncbi_tax_id(tax: Any) -> int | None:
    for entry in _as_list(tax):
        if isinstance(entry, dict):
            value = entry.get("NCBI tax id")
            if value is not None:
                try:
                    return int(value)
                except (ValueError, TypeError):
                    continue
    return None
