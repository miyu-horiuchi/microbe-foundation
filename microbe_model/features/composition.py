"""Compositional features: k-mer frequencies and codon usage.

These supplement the v0 amino-acid-composition features in `genome.py`. They are
computed on the same predicted-CDS set, so adding them to a v1 featurize run is
~free in network/CPU terms.

Two feature groups:
  - tetranucleotide frequencies (256 dims) — well-known signal for thermophily,
    halophily, and phylum-level taxonomy
  - codon usage frequencies (64 dims) — informs translation efficiency, GC bias,
    and growth rate phenotype

We use them as relative frequencies (sum to 1 across each group) rather than
counts, so they're scale-invariant across genome sizes.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

NUCLEOTIDES = "ACGT"
TETRA_KMERS = [a + b + c + d for a in NUCLEOTIDES for b in NUCLEOTIDES
               for c in NUCLEOTIDES for d in NUCLEOTIDES]
CODONS = [a + b + c for a in NUCLEOTIDES for b in NUCLEOTIDES for c in NUCLEOTIDES]


def tetranucleotide_freqs(contigs: Iterable[tuple[str, str]]) -> dict[str, float]:
    """Relative frequency of each of the 256 ACGT tetranucleotides.

    Skips any 4-mer containing a non-ACGT character (e.g. N).
    """
    counts: Counter[str] = Counter()
    total = 0
    for _, seq in contigs:
        s = seq.upper()
        for i in range(len(s) - 3):
            kmer = s[i : i + 4]
            if kmer in TETRA_KMERS_SET:  # fast in-set check
                counts[kmer] += 1
                total += 1
    if total == 0:
        return {f"tetra_{k}": 0.0 for k in TETRA_KMERS}
    return {f"tetra_{k}": counts.get(k, 0) / total for k in TETRA_KMERS}


def codon_freqs(cds_nucleotides: Iterable[str]) -> dict[str, float]:
    """Relative frequency of each of the 64 codons across all predicted CDS.

    Argument: an iterable of nucleotide CDS strings (multiples of 3, ATG-start).
    Skips codons containing non-ACGT (e.g. N).
    """
    counts: Counter[str] = Counter()
    total = 0
    for cds in cds_nucleotides:
        s = cds.upper()
        for i in range(0, len(s) - 2, 3):
            codon = s[i : i + 3]
            if codon in CODONS_SET:
                counts[codon] += 1
                total += 1
    if total == 0:
        return {f"codon_{k}": 0.0 for k in CODONS}
    return {f"codon_{k}": counts.get(k, 0) / total for k in CODONS}


# Lookup sets for fast membership checks
TETRA_KMERS_SET = set(TETRA_KMERS)
CODONS_SET = set(CODONS)
