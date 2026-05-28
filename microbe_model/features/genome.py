"""Tabular feature extraction from a microbial genome FASTA.

These features are deliberately simple and biologically motivated:
  - genome size, GC content, coding density
  - predicted gene count and mean CDS length
  - proteome-level amino acid composition
  - aromatic, charged, and IVYWREL fractions (correlate with growth temperature)
  - mean isoelectric point and hydrophobicity

The amino-acid-composition signals have well-established correlations with optimal growth
temperature and pH (Zeldovich 2007; Tekaia 2002), so they give XGBoost real signal to learn from
without any deep model.
"""
from __future__ import annotations

import gzip
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pyrodigal
from Bio import SeqIO

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_AROMATIC = set("FWY")
AA_CHARGED_POS = set("KRH")
AA_CHARGED_NEG = set("DE")
AA_IVYWREL = set("IVYWREL")  # thermophile signature (Zeldovich 2007)

# Kyte-Doolittle hydrophobicity
HYDROPHOBICITY = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4, "H": -3.2,
    "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5,
    "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

# pKa values for isoelectric point estimation (Lehninger)
PKA_NTERM = 9.69
PKA_CTERM = 2.34
PKA_SIDE = {"D": 3.65, "E": 4.25, "C": 8.33, "Y": 10.07, "H": 6.00, "K": 10.53, "R": 12.48}


def read_fasta_records(path: Path) -> Iterable[tuple[str, str]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            yield record.id, str(record.seq).upper()


MIN_TRAIN_NT = 20_000  # below this, pyrodigal can't train; fall back to meta mode


def predict_genes(contigs: Iterable[tuple[str, str]]) -> tuple[list[str], list[str], int]:
    """Run Pyrodigal and return (proteins, nt_cds_sequences, total_nt).

    Uses single-genome mode with training on the concatenated contigs — ~7× faster than
    meta mode on assembled genomes. Falls back to meta mode for very short or highly
    fragmented assemblies that can't be trained.
    """
    contigs = list(contigs)  # we need to traverse twice
    encoded = [(name, seq.encode("ascii")) for name, seq in contigs]
    total_nt = sum(len(seq) for _, seq in encoded)

    if total_nt >= MIN_TRAIN_NT:
        finder = pyrodigal.GeneFinder(meta=False)
        train_seq = b"TTAATTAATTAA".join(seq for _, seq in encoded)
        try:
            finder.train(train_seq)
        except Exception:
            finder = pyrodigal.GeneFinder(meta=True)
    else:
        finder = pyrodigal.GeneFinder(meta=True)

    proteins: list[str] = []
    cds: list[str] = []
    for _name, seq in encoded:
        genes = finder.find_genes(seq)
        for gene in genes:
            proteins.append(gene.translate().rstrip("*"))
            cds.append(gene.sequence())
    return proteins, cds, total_nt


def predict_proteins(contigs: Iterable[tuple[str, str]]) -> tuple[list[str], int]:
    """Backwards-compat shim — returns (proteins, total_nt) only."""
    proteins, _cds, total_nt = predict_genes(contigs)
    return proteins, total_nt


def aa_composition(proteins: list[str]) -> dict[str, float]:
    counts: Counter[str] = Counter()
    total = 0
    for p in proteins:
        counts.update(p)
        total += len(p)
    if total == 0:
        return {f"aa_frac_{a}": 0.0 for a in AA_ALPHABET}
    return {f"aa_frac_{a}": counts.get(a, 0) / total for a in AA_ALPHABET}


def _isoelectric_point(seq: str) -> float:
    """Bisection over pH to find the point where net charge is zero."""
    if not seq:
        return 7.0
    counts = Counter(seq)
    lo, hi = 0.0, 14.0
    for _ in range(50):
        ph = (lo + hi) / 2
        net = (
            1 / (1 + 10 ** (ph - PKA_NTERM))
            - 1 / (1 + 10 ** (PKA_CTERM - ph))
            + counts.get("K", 0) / (1 + 10 ** (ph - PKA_SIDE["K"]))
            + counts.get("R", 0) / (1 + 10 ** (ph - PKA_SIDE["R"]))
            + counts.get("H", 0) / (1 + 10 ** (ph - PKA_SIDE["H"]))
            - counts.get("D", 0) / (1 + 10 ** (PKA_SIDE["D"] - ph))
            - counts.get("E", 0) / (1 + 10 ** (PKA_SIDE["E"] - ph))
            - counts.get("C", 0) / (1 + 10 ** (PKA_SIDE["C"] - ph))
            - counts.get("Y", 0) / (1 + 10 ** (PKA_SIDE["Y"] - ph))
        )
        if net > 0:
            lo = ph
        else:
            hi = ph
    return (lo + hi) / 2


def extract_features_from_seqs(
    contigs: list[tuple[str, str]],
    *,
    include_composition: bool = True,
) -> dict[str, float]:
    """Compute the full feature dict given pre-loaded contigs.

    Used by the streaming pipeline to avoid round-tripping FASTA bytes through disk.
    When ``include_composition`` is True (default), tetranucleotide and codon-usage
    features are appended (320 extra columns).
    """
    nt_total = sum(len(s) for _, s in contigs)
    gc = sum(s.count("G") + s.count("C") for _, s in contigs)
    gc_frac = gc / nt_total if nt_total else 0.0

    proteins, cds, _ = predict_genes(contigs)
    aa_total = sum(len(p) for p in proteins)
    coding_density = (3 * aa_total) / nt_total if nt_total else 0.0

    composition = aa_composition(proteins)

    aromatic = sum(composition[f"aa_frac_{a}"] for a in AA_AROMATIC)
    pos_charged = sum(composition[f"aa_frac_{a}"] for a in AA_CHARGED_POS)
    neg_charged = sum(composition[f"aa_frac_{a}"] for a in AA_CHARGED_NEG)
    ivywrel = sum(composition[f"aa_frac_{a}"] for a in AA_IVYWREL)

    hydrophobicity = (
        sum(composition[f"aa_frac_{a}"] * HYDROPHOBICITY[a] for a in AA_ALPHABET)
        if proteins else 0.0
    )

    pi_values = [_isoelectric_point(p) for p in proteins[:200]]  # 200 sampled proteins is plenty
    mean_pi = float(np.mean(pi_values)) if pi_values else 7.0

    cds_lengths = [len(p) for p in proteins]
    feats: dict[str, float] = {
        "genome_size_nt": float(nt_total),
        "n_contigs": float(len(contigs)),
        "gc_content": gc_frac,
        "n_predicted_cds": float(len(proteins)),
        "coding_density": coding_density,
        "mean_cds_aa_length": float(np.mean(cds_lengths)) if cds_lengths else 0.0,
        "median_cds_aa_length": float(np.median(cds_lengths)) if cds_lengths else 0.0,
        "aromatic_frac": aromatic,
        "pos_charged_frac": pos_charged,
        "neg_charged_frac": neg_charged,
        "ivywrel_frac": ivywrel,
        "mean_hydrophobicity": hydrophobicity,
        "mean_isoelectric_point": mean_pi,
        **composition,
    }
    if include_composition:
        from microbe_model.features.composition import codon_freqs, tetranucleotide_freqs
        feats.update(tetranucleotide_freqs(contigs))
        feats.update(codon_freqs(cds))
    return feats


def extract_features(fasta_path: Path) -> dict[str, float]:
    """Disk-based entry point — convenience wrapper for non-streaming use."""
    return extract_features_from_seqs(list(read_fasta_records(fasta_path)))
