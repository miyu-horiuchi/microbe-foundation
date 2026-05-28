"""
extract_genome_accessions.py — pull NCBI assembly accessions for strains
with genome sequences in BacDive.

Reads:  data/bacdive_raw.jsonl
Writes: data/genome_accessions.tsv  with columns:
    bacdive_id, accession, assembly_level, score, description

For strains with multiple genomes listed, picks the highest-scoring
assembly (BacDive's score reflects assembly quality / type-strain match).
'complete' assemblies are preferred over 'scaffold' / 'contig' on ties.

Output feeds NCBI Datasets CLI or any other genome downloader:
    datasets download genome accession $(cut -f2 genome_accessions.tsv | tail -n+2 | head -1000) \\
        --include genome --filename batch1.zip

Usage:
    python extract_genome_accessions.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_IN = DATA_DIR / "bacdive_raw.jsonl"
DEFAULT_OUT = DATA_DIR / "genome_accessions.tsv"


ASSEMBLY_LEVEL_RANK = {"complete": 4, "chromosome": 3, "scaffold": 2, "contig": 1}


def best_genome(genomes: list[dict]) -> dict | None:
    """Pick the highest-scoring assembly; tie-break by assembly level."""
    valid = [g for g in genomes if isinstance(g, dict) and g.get("INSDC accession")]
    if not valid:
        return None
    return max(
        valid,
        key=lambda g: (
            float(g.get("score", 0)) if g.get("score") is not None else 0.0,
            ASSEMBLY_LEVEL_RANK.get(str(g.get("assembly level", "")).lower(), 0),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="input_path", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", dest="output_path", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.input_path.exists():
        sys.exit(f"Input file not found: {args.input_path}")

    args.output_path.parent.mkdir(exist_ok=True)
    n_total = n_with_genome = 0
    level_counts: dict[str, int] = {}

    with args.output_path.open("w") as out_f:
        out_f.write("bacdive_id\taccession\tassembly_level\tscore\tdescription\n")
        with args.input_path.open() as in_f:
            for line in in_f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_total += 1
                bid = r.get("_bacdive_id")
                seq = r.get("Sequence information", {}) or {}
                if not isinstance(seq, dict):
                    continue
                genomes = seq.get("Genome sequences")
                if genomes is None:
                    continue
                glist = genomes if isinstance(genomes, list) else [genomes]
                best = best_genome(glist)
                if best is None:
                    continue
                n_with_genome += 1
                level = str(best.get("assembly level", "")).lower() or "unknown"
                level_counts[level] = level_counts.get(level, 0) + 1
                desc = str(best.get("description", "")).replace("\t", " ").replace("\n", " ")[:200]
                out_f.write(
                    f"{bid}\t{best['INSDC accession']}\t{level}\t"
                    f"{best.get('score', '')}\t{desc}\n"
                )

    print(f"Processed {n_total:,} BacDive records")
    print(f"  with at least one genome: {n_with_genome:,} ({100*n_with_genome/max(n_total,1):.1f}%)")
    print(f"  by assembly level:")
    for lvl, c in sorted(level_counts.items(), key=lambda x: -x[1]):
        print(f"    {lvl:<12} {c:>6,}")
    print(f"\nWrote {n_with_genome:,} accessions to {args.output_path}")


if __name__ == "__main__":
    main()
