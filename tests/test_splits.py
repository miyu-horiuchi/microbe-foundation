"""Split correctness — the benchmark's most critical invariant."""
from __future__ import annotations

import splits as splits_mod


def test_no_group_spans_buckets():
    """Whole groups must never split across buckets — this is the benchmark's defining property."""
    sizes = {f"fam_{i}": 100 for i in range(50)}
    assignment = splits_mod.assign_groups(sizes, val_frac=0.1, test_frac=0.1, seed=0)
    # Each family gets one bucket — invariant by construction
    assert len(assignment) == len(sizes)
    assert set(assignment.values()) <= {"train", "val", "test"}


def test_hits_target_ratios_at_scale():
    """At scale (many small uniform groups), splits should hit 80/10/10 closely."""
    sizes = {f"g_{i}": 100 for i in range(500)}
    assignment = splits_mod.assign_groups(sizes, val_frac=0.1, test_frac=0.1, seed=0)
    total = sum(sizes.values())
    by_bucket = {"train": 0, "val": 0, "test": 0}
    for name, size in sizes.items():
        by_bucket[assignment[name]] += size
    # Exact 80/10/10 — uniform sizes + largest-first greedy
    assert abs(by_bucket["train"] / total - 0.80) < 0.001
    assert abs(by_bucket["val"] / total - 0.10) < 0.001
    assert abs(by_bucket["test"] / total - 0.10) < 0.001


def test_handles_lopsided_group_sizes():
    """A handful of huge groups + many tiny ones should still produce balanced strain counts."""
    sizes = {"giant_a": 1000, "giant_b": 1000, "giant_c": 1000}
    for i in range(100):
        sizes[f"tiny_{i}"] = 10
    assignment = splits_mod.assign_groups(sizes, val_frac=0.1, test_frac=0.1, seed=42)
    total = sum(sizes.values())
    by_bucket = {"train": 0, "val": 0, "test": 0}
    for name, size in sizes.items():
        by_bucket[assignment[name]] += size
    # Lopsided but largest-first packing keeps strain-count ratios sane
    test_frac = by_bucket["test"] / total
    val_frac = by_bucket["val"] / total
    assert 0.05 <= test_frac <= 0.20, f"test fraction {test_frac:.3f} way off target"
    assert 0.05 <= val_frac <= 0.20, f"val fraction {val_frac:.3f} way off target"


def test_seed_determinism():
    """Same seed → identical split. Different seed → different split.

    Use uniform sizes so the sort doesn't dominate — only the shuffle (seeded)
    decides which equal-sized group lands in which bucket.
    """
    sizes = {f"g_{i}": 100 for i in range(50)}
    a1 = splits_mod.assign_groups(sizes, 0.1, 0.1, seed=0)
    a2 = splits_mod.assign_groups(sizes, 0.1, 0.1, seed=0)
    a3 = splits_mod.assign_groups(sizes, 0.1, 0.1, seed=99)
    assert a1 == a2, "same seed should be deterministic"
    assert a1 != a3, "different seed should produce a different assignment"


def test_empty_input():
    """Empty input is a no-op, not a crash."""
    assert splits_mod.assign_groups({}, 0.1, 0.1, seed=0) == {}
