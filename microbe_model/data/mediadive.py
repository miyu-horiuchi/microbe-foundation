"""MediaDive (DSMZ) integration — strain↔medium links and full recipes.

The BacDive v2 records we already cached include inline medium links of the form
``https://mediadive.dsmz.de/medium/{id}`` plus a `growth: yes/no` flag. So extracting
strain↔medium pairs needs no new API calls. The medium *recipes* (compound list
with amounts) do need network access via MediaDive's REST API.

API documentation observed live on 2026-04-27:
  - /rest/medium/{id}   → full recipe with solutions[].recipe[] (compound + amount + unit + g_l)
  - /rest/media         → paginated list of all media (limit + offset)
  - /rest/medium-strains/{id} → strains linked to a medium (with bacdive_id)
"""
from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import requests

from microbe_model import config

BASE_URL = "https://mediadive.dsmz.de/rest"
RATE_LIMIT_S = 0.3  # be polite to a small public API


def _extract_medium_id(link: str | None) -> str | None:
    if not link:
        return None
    m = re.search(r"/medium/([A-Za-z0-9]+)", link)
    return m.group(1) if m else None


def parse_strain_media_links(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a list of {medium_id, medium_name, growth} for each medium in a BacDive record."""
    culture = record.get("Culture and growth conditions") or {}
    raw = culture.get("culture medium") or []
    if isinstance(raw, dict):
        raw = [raw]

    out: list[dict[str, Any]] = []
    for m in raw:
        if not isinstance(m, dict):
            continue
        medium_id = _extract_medium_id(m.get("link"))
        if not medium_id:
            continue
        growth = (m.get("growth") or "").strip().lower()
        out.append({
            "medium_id": str(medium_id),
            "medium_name": m.get("name"),
            "growth": growth,  # "yes", "no", "weak", or ""
        })
    return out


def iter_bacdive_strain_media(cache_dir: Path | None = None) -> Iterator[dict[str, Any]]:
    """Walk the BacDive cache and yield {bacdive_id, medium_id, medium_name, growth} rows."""
    cache_dir = cache_dir or config.BACDIVE_DIR
    for path in cache_dir.glob("*.json"):
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        try:
            bid = int(path.stem)
        except ValueError:
            continue
        for link in parse_strain_media_links(record):
            yield {
                "bacdive_id": bid,
                "medium_id": link["medium_id"],
                "medium_name": link["medium_name"],
                "growth": link["growth"],
            }


class MediaDiveClient:
    """Polite REST client for MediaDive — 0.3s sleep between calls by default."""

    def __init__(self, *, rate_limit_s: float = RATE_LIMIT_S) -> None:
        self.session = requests.Session()
        self.rate_limit_s = rate_limit_s

    def _get(self, path: str, params: dict | None = None) -> dict[str, Any]:
        time.sleep(self.rate_limit_s)
        url = f"{BASE_URL}{path}"
        for attempt in range(3):
            try:
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code in (429, 502, 503):
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        return {}

    def fetch_medium(self, medium_id: str) -> dict[str, Any] | None:
        """Return the full medium record, or None if not found / malformed."""
        try:
            body = self._get(f"/medium/{medium_id}")
        except requests.HTTPError:
            return None
        if body.get("status") != 200:
            return None
        return body.get("data") or None

    def list_media(self, *, limit: int = 200, offset: int = 0) -> list[dict[str, Any]]:
        body = self._get("/media", params={"limit": limit, "offset": offset})
        return body.get("data") or []


def normalize_recipe(medium_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a /medium/{id} payload into per-compound rows.

    Each row: {medium_id, solution_name, compound_id, compound, amount, unit, g_l, optional}.
    Skips compounds with no g_l / amount.
    """
    medium = medium_payload.get("medium") or {}
    medium_id = str(medium.get("id", ""))
    rows: list[dict[str, Any]] = []
    for solution in medium_payload.get("solutions") or []:
        sol_name = solution.get("name", "")
        for r in solution.get("recipe") or []:
            if not isinstance(r, dict):
                continue
            compound = r.get("compound")
            if not compound:
                continue
            rows.append({
                "medium_id": medium_id,
                "solution_name": sol_name,
                "compound_id": r.get("compound_id"),
                "compound": compound,
                "amount": r.get("amount"),
                "unit": r.get("unit"),
                "g_l": r.get("g_l"),
                "optional": int(r.get("optional", 0) or 0),
                "condition": r.get("condition"),
            })
    return rows
