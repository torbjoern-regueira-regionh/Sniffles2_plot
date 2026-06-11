# -*- coding: utf-8 -*-
import collections
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from sniffles2_plot.helper.io_class import open_vcf
from sniffles2_plot.parser.vcf_line_parser import VCFLineSV


def _load_sv_buckets(vcf_path: str, slop: int) -> set:
    """Return a set of (svtype, chrom, bucket) for one VCF file."""
    buckets = set()
    with open_vcf(vcf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            obj = VCFLineSV(line)
            if obj.ERROR or not obj.SVTYPE or obj.POS is None:
                continue
            bucket = obj.POS // slop
            buckets.add((obj.SVTYPE, obj.CHROM, bucket))
    return buckets


def _upset_plot(membership_counts: dict, caller_names: list, title: str, out_path: str):
    """
    Draw an upset plot from scratch using matplotlib.

    membership_counts: {frozenset_of_caller_names: count}
    """
    if not membership_counts:
        print(f"No data for '{title}', skipping.")
        return

    n_callers = len(caller_names)

    # Sort intersections by count descending
    sorted_intersections = sorted(membership_counts.items(), key=lambda x: -x[1])
    n_intersections = len(sorted_intersections)

    counts = [c for _, c in sorted_intersections]
    memberships = [m for m, _ in sorted_intersections]

    fig = plt.figure(figsize=(max(8, n_intersections * 0.6 + 2), 4 + n_callers * 0.5))
    gs = gridspec.GridSpec(
        2, 1,
        height_ratios=[3, n_callers * 0.5],
        hspace=0.05,
    )

    # --- Top: bar chart of intersection sizes ---
    ax_bar = fig.add_subplot(gs[0])
    x = np.arange(n_intersections)
    bars = ax_bar.bar(x, counts, color="black", width=0.5)
    for bar, count in zip(bars, counts):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center", va="bottom", fontsize=7,
        )
    ax_bar.set_ylabel("Intersection size")
    ax_bar.set_xticks([])
    ax_bar.set_title(title, fontsize=9)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # --- Bottom: dot matrix ---
    ax_dot = fig.add_subplot(gs[1])
    ax_dot.set_xlim(-0.5, n_intersections - 0.5)
    ax_dot.set_ylim(-0.5, n_callers - 0.5)

    for col_idx, member_set in enumerate(memberships):
        active = [i for i, name in enumerate(caller_names) if name in member_set]
        inactive = [i for i, name in enumerate(caller_names) if name not in member_set]
        # filled dots for active callers
        ax_dot.scatter([col_idx] * len(active), active, color="black", s=80, zorder=3)
        # empty dots for inactive
        ax_dot.scatter([col_idx] * len(inactive), inactive,
                       color="lightgrey", s=80, zorder=3)
        # vertical line connecting active dots
        if len(active) > 1:
            ax_dot.plot([col_idx, col_idx], [min(active), max(active)],
                        color="black", lw=2, zorder=2)

    ax_dot.set_yticks(range(n_callers))
    ax_dot.set_yticklabels(caller_names, fontsize=8)
    ax_dot.set_xticks([])
    ax_dot.spines["top"].set_visible(False)
    ax_dot.spines["right"].set_visible(False)
    ax_dot.spines["bottom"].set_visible(False)

    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def generate_caller_comparison_upset(vcf_dir: str, output_dir: str, slop: int = 500):
    """
    Read all VCF files in vcf_dir (one per caller).  Cluster SVs by
    (SVTYPE, CHROM, POS // slop) and produce one upset plot per SVTYPE
    plus a combined plot, showing which callers share each call.
    """
    vcf_files = sorted(
        e.path
        for e in os.scandir(vcf_dir)
        if e.name.lower().endswith(".vcf") or e.name.lower().endswith(".vcf.gz")
    )
    if len(vcf_files) < 2:
        print("Caller comparison requires at least 2 VCF files in the input directory.")
        return

    def _stem(path):
        name = os.path.basename(path)
        for ext in (".vcf.gz", ".vcf"):
            if name.lower().endswith(ext):
                name = name[: -len(ext)]
                break
        return name

    caller_names = [_stem(p) for p in vcf_files]
    print(f"Comparing callers: {', '.join(caller_names)}")

    # caller_buckets[i] = set of (svtype, chrom, bucket)
    caller_buckets = [_load_sv_buckets(p, slop) for p in vcf_files]

    # Universe: every unique key seen by at least one caller
    universe = set().union(*caller_buckets)

    # key -> frozenset of caller names that found it
    key_to_callers: dict = {}
    for key in universe:
        members = frozenset(
            name for name, buckets in zip(caller_names, caller_buckets) if key in buckets
        )
        key_to_callers[key] = members

    os.makedirs(output_dir, exist_ok=True)

    # Group keys by SVTYPE
    keys_by_svtype: dict = collections.defaultdict(list)
    for key in universe:
        keys_by_svtype[key[0]].append(key)

    for svtype, keys in sorted(keys_by_svtype.items()):
        counts: dict = collections.Counter(key_to_callers[k] for k in keys)
        _upset_plot(
            counts,
            caller_names,
            f"Caller comparison — {svtype}  (window={slop} bp)",
            os.path.join(output_dir, f"caller_upset_{svtype}.jpg"),
        )

    # Combined across all SV types
    all_counts: dict = collections.Counter(key_to_callers[k] for k in universe)
    _upset_plot(
        all_counts,
        caller_names,
        f"Caller comparison — all SV types  (window={slop} bp)",
        os.path.join(output_dir, "caller_upset_combined.jpg"),
    )
