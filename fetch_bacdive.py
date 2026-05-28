"""
fetch_bacdive.py — bulk-fetch BacDive strain records via the public v2 API.

The BacDive v2 API at api.bacdive.dsmz.de/v2/fetch/{id} is publicly accessible
without authentication as of Feb 2026 (CC-BY 4.0). Be respectful — modest
concurrency, back off on 429/5xx.

Output:
    data/bacdive_raw.jsonl    one strain record per line, with `_bacdive_id` prefixed
    data/bacdive_done.txt     completed IDs, one per line (for resumability)

Both files are appended to, so re-running the script picks up where it left off.

Usage:
    python fetch_bacdive.py                          # fetch IDs 1..200000
    python fetch_bacdive.py --start 1 --end 1000     # smoke-test range
    python fetch_bacdive.py --workers 10 --sleep 0.1
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

API_BASE = "https://api.bacdive.dsmz.de/v2/fetch"
USER_AGENT = (
    "microbe-foundation/0.1.0 (research project; "
    "https://github.com/miyu-horiuchi/microbe-foundation)"
)

DATA_DIR = Path(__file__).parent / "data"
RAW_PATH = DATA_DIR / "bacdive_raw.jsonl"
DONE_PATH = DATA_DIR / "bacdive_done.txt"


def fetch_one(
    bacdive_id: int, timeout: float = 30.0, max_retries: int = 5
) -> tuple[int, dict | None, str]:
    """Fetch one strain. Returns (id, record_dict_or_None, status_str)."""
    url = f"{API_BASE}/{bacdive_id}"
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})

    for attempt in range(max_retries):
        try:
            with urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    # API wraps strain data as {count, next, previous, results: {str(id): {...}}}
                    # Unwrap to just the inner strain record.
                    wrapper = json.loads(resp.read())
                    inner = wrapper.get("results", {}).get(str(bacdive_id))
                    if inner is None:
                        return (bacdive_id, None, "empty_results")
                    return (bacdive_id, inner, "ok")
                return (bacdive_id, None, f"http_{resp.status}")
        except HTTPError as e:
            if e.code == 404:
                return (bacdive_id, None, "404")
            if e.code == 429 or 500 <= e.code < 600:
                time.sleep(min(2**attempt, 30))
                continue
            return (bacdive_id, None, f"http_{e.code}")
        except (URLError, TimeoutError, ConnectionError):
            time.sleep(min(2**attempt, 30))
        except json.JSONDecodeError:
            return (bacdive_id, None, "json_error")

    return (bacdive_id, None, "max_retries_exceeded")


def load_done(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open() as f:
        return {int(line.strip()) for line in f if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=int, default=1, help="First BacDive ID (inclusive). Default: 1")
    parser.add_argument("--end", type=int, default=200_000, help="Last BacDive ID (inclusive). Default: 200000")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent HTTP workers. Default: 10")
    parser.add_argument("--batch", type=int, default=200, help="IDs per progress-report batch. Default: 200")
    parser.add_argument("--sleep", type=float, default=0.1, help="Sleep seconds between batches. Default: 0.1")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    done = load_done(DONE_PATH)
    todo = [i for i in range(args.start, args.end + 1) if i not in done]

    print("BacDive bulk fetcher")
    print(f"  range:        {args.start:,}..{args.end:,}  ({args.end - args.start + 1:,} IDs)")
    print(f"  already done: {len(done):,}")
    print(f"  to fetch:     {len(todo):,}")
    print(f"  workers:      {args.workers}")
    print(f"  output:       {RAW_PATH}")
    print()

    if not todo:
        print("Nothing to fetch — exiting.")
        return

    n_ok = n_404 = n_err = 0
    start_time = time.time()

    with RAW_PATH.open("a") as raw_f, DONE_PATH.open("a") as done_f, ThreadPoolExecutor(
        max_workers=args.workers
    ) as ex:
        for batch_start in range(0, len(todo), args.batch):
            batch = todo[batch_start : batch_start + args.batch]
            futures = [ex.submit(fetch_one, bid) for bid in batch]

            for fut in as_completed(futures):
                bid, record, status = fut.result()
                done_f.write(f"{bid}\n")

                if status == "ok" and record is not None:
                    raw_f.write(json.dumps({"_bacdive_id": bid, **record}) + "\n")
                    n_ok += 1
                elif status == "404":
                    n_404 += 1
                else:
                    n_err += 1

            raw_f.flush()
            done_f.flush()

            elapsed = time.time() - start_time
            processed = n_ok + n_404 + n_err
            rate = processed / elapsed if elapsed > 0 else 0.0
            eta_min = (len(todo) - processed) / rate / 60 if rate > 0 else 0.0
            print(
                f"  [{processed:>7,}/{len(todo):,}]  "
                f"ok={n_ok:,}  404={n_404:,}  err={n_err:,}  "
                f"rate={rate:5.1f}/s  eta={eta_min:5.1f}min"
            )

            time.sleep(args.sleep)

    print()
    print(f"Done. {n_ok:,} records saved to {RAW_PATH}")
    print(f"      {n_404:,} unassigned IDs (expected — most BacDive IDs are sparse)")
    print(f"      {n_err:,} errors")


if __name__ == "__main__":
    main()
