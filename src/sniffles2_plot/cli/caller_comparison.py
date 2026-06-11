# -*- coding: utf-8 -*-
import collections
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from sniffles2_plot.helper.io_class import open_vcf
from sniffles2_plot.parser.vcf_line_parser import VCFLineSV

# Colour palette for SVTYPE in the genome scatter plot
_SVTYPE_COLORS = {
    "DEL": "#d62728",
    "INS": "#1f77b4",
    "DUP": "#2ca02c",
    "INV": "#ff7f0e",
    "BND": "#9467bd",
    "CNV": "#8c564b",
}
_DEFAULT_COLOR = "#7f7f7f"

# Natural chromosome sort order
_CHR_ORDER = (
    [f"chr{i}" for i in range(1, 23)]
    + ["chrX", "chrY", "chrM", "chrMT"]
    + [str(i) for i in range(1, 23)]
    + ["X", "Y", "MT", "M"]
)


def _chr_sort_key(chrom: str) -> tuple:
    try:
        return (0, _CHR_ORDER.index(chrom))
    except ValueError:
        return (1, chrom)


def _load_sv_buckets(vcf_path: str, slop: int) -> tuple:
    """
    Return (buckets, repr_info) where:
      buckets  : set of (svtype, chrom, bucket)
      repr_info: dict (svtype, chrom, bucket) -> (chrom, pos, svtype, svlen)
    """
    buckets = set()
    repr_info = {}
    with open_vcf(vcf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            obj = VCFLineSV(line)
            if obj.ERROR or not obj.SVTYPE or obj.POS is None:
                continue
            bucket = obj.POS // slop
            key = (obj.SVTYPE, obj.CHROM, bucket)
            buckets.add(key)
            if key not in repr_info:
                svlen = abs(obj.SVLEN) if obj.SVLEN else 0
                repr_info[key] = (obj.CHROM, obj.POS, obj.SVTYPE, svlen)
    return buckets, repr_info


def _upset_plot(
    membership_counts: dict,
    caller_names: list,
    title: str,
    out_path: str,
    multicaller_variants: list = None,
    min_callers: int = None,
    table_limit: int = 50,
):
    """
    Draw an upset plot.

    membership_counts   : {frozenset_of_caller_names: count}
    multicaller_variants: list of (chrom, pos, svtype, svlen, n_callers) — if provided
                          and len <= table_limit, a summary table is appended to the figure.
    """
    if not membership_counts:
        print(f"No data for '{title}', skipping.")
        return

    n_callers = len(caller_names)
    show_table = (
        multicaller_variants is not None
        and 0 < len(multicaller_variants) <= table_limit
    )

    sorted_intersections = sorted(membership_counts.items(), key=lambda x: -x[1])
    n_intersections = len(sorted_intersections)
    counts = [c for _, c in sorted_intersections]
    memberships = [m for m, _ in sorted_intersections]

    n_rows = 2 + (1 if show_table else 0)
    height_ratios = [3, max(1, n_callers * 0.5)]
    if show_table:
        table_rows = min(len(multicaller_variants), table_limit)
        height_ratios.append(max(1.5, table_rows * 0.25))

    fig = plt.figure(figsize=(max(8, n_intersections * 0.6 + 2),
                               sum(height_ratios) + 1))
    gs = gridspec.GridSpec(n_rows, 1, height_ratios=height_ratios, hspace=0.1)

    # --- Top: bar chart ---
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

    # --- Middle: dot matrix ---
    ax_dot = fig.add_subplot(gs[1])
    ax_dot.set_xlim(-0.5, n_intersections - 0.5)
    ax_dot.set_ylim(-0.5, n_callers - 0.5)
    for col_idx, member_set in enumerate(memberships):
        active = [i for i, name in enumerate(caller_names) if name in member_set]
        inactive = [i for i, name in enumerate(caller_names) if name not in member_set]
        ax_dot.scatter([col_idx] * len(active), active, color="black", s=80, zorder=3)
        ax_dot.scatter([col_idx] * len(inactive), inactive, color="lightgrey", s=80, zorder=3)
        if len(active) > 1:
            ax_dot.plot([col_idx, col_idx], [min(active), max(active)],
                        color="black", lw=2, zorder=2)
    ax_dot.set_yticks(range(n_callers))
    ax_dot.set_yticklabels(caller_names, fontsize=8)
    ax_dot.set_xticks([])
    ax_dot.spines["top"].set_visible(False)
    ax_dot.spines["right"].set_visible(False)
    ax_dot.spines["bottom"].set_visible(False)

    # --- Bottom: table of multi-caller variants ---
    if show_table:
        ax_tbl = fig.add_subplot(gs[2])
        ax_tbl.axis("off")
        rows = sorted(multicaller_variants, key=lambda v: (_chr_sort_key(v[0]), v[1]))
        table_data = [
            [v[0], str(v[1]), v[2], str(v[3]) if v[3] else ".", str(v[4])]
            for v in rows
        ]
        col_labels = ["CHROM", "POS", "SVTYPE", "SVLEN", f"#callers (≥{min_callers})"]
        tbl = ax_tbl.table(
            cellText=table_data,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(6)
        tbl.auto_set_column_width(col=list(range(len(col_labels))))
        ax_tbl.set_title(
            f"Variants reported by ≥{min_callers} callers (n={len(multicaller_variants)})",
            fontsize=7, pad=2,
        )

    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def _genome_scatter_plot(
    variants: list,
    caller_names: list,
    title: str,
    out_path: str,
):
    """
    Genome-wide scatter of multi-caller variants.
    variants: list of (chrom, pos, svtype, svlen, n_callers)
    """
    chroms = sorted({v[0] for v in variants}, key=_chr_sort_key)
    chrom_index = {c: i for i, c in enumerate(chroms)}
    n_callers = len(caller_names)

    fig, ax = plt.subplots(figsize=(max(10, len(chroms) * 0.6), 6))

    for v in variants:
        chrom, pos, svtype, svlen, n = v
        xi = chrom_index[chrom]
        color = _SVTYPE_COLORS.get(svtype, _DEFAULT_COLOR)
        # Dot size scales with number of callers
        size = 30 + (n / n_callers) * 120
        ax.scatter(xi, pos, c=color, s=size, alpha=0.6, edgecolors="none", zorder=2)

    ax.set_xticks(range(len(chroms)))
    ax.set_xticklabels(chroms, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Genomic position (bp)")
    ax.set_title(title, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend for SVTYPE colours
    seen_types = sorted({v[2] for v in variants})
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=_SVTYPE_COLORS.get(t, _DEFAULT_COLOR),
                   markersize=6, label=t)
        for t in seen_types
    ]
    # Size legend for caller count
    for n in sorted({v[4] for v in variants}):
        s = 30 + (n / n_callers) * 120
        handles.append(
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor="grey", alpha=0.6,
                       markersize=np.sqrt(s) * 0.5,
                       label=f"{n} caller{'s' if n > 1 else ''}")
        )
    ax.legend(handles=handles, loc="upper right", fontsize=7, frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def _write_tsv(variants: list, caller_names: list, out_path: str, min_callers: int):
    """Write multi-caller variants to a TSV file."""
    with open(out_path, "w") as f:
        f.write("CHROM\tPOS\tSVTYPE\tSVLEN\tN_CALLERS\tCALLERS\n")
        for v in sorted(variants, key=lambda x: (_chr_sort_key(x[0]), x[1])):
            chrom, pos, svtype, svlen, n = v
            # We only have n_callers count here; callers string is added separately
            f.write(f"{chrom}\t{pos}\t{svtype}\t{svlen if svlen else '.'}\t{n}\t.\n")
    print(f"Saved: {out_path}")


def generate_caller_comparison_upset(
    vcf_dir: str,
    output_dir: str,
    slop: int = 500,
    min_callers: int = None,
    table_limit: int = 50,
):
    """
    Read all VCF files in vcf_dir (one per caller).  Cluster SVs by
    (SVTYPE, CHROM, POS // slop) and produce one upset plot per SVTYPE
    plus a combined plot.

    If min_callers is set, variants found by >= min_callers callers are:
      - Always written to a TSV file.
      - Embedded as a table on the upset plot if count <= table_limit.
      - Shown as a genome scatter plot if count > table_limit.
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

    # Load buckets and representative coordinates per caller
    all_buckets = []
    all_repr = {}  # merged repr_info across all callers (first-seen wins)
    for p in vcf_files:
        buckets, repr_info = _load_sv_buckets(p, slop)
        all_buckets.append(buckets)
        for k, v in repr_info.items():
            if k not in all_repr:
                all_repr[k] = v

    universe = set().union(*all_buckets)

    # key -> frozenset of caller names
    key_to_callers: dict = {
        key: frozenset(
            name for name, buckets in zip(caller_names, all_buckets) if key in buckets
        )
        for key in universe
    }

    os.makedirs(output_dir, exist_ok=True)

    def _multicaller_variants_for(keys):
        """Return list of (chrom, pos, svtype, svlen, n_callers) for keys with >= min_callers."""
        if min_callers is None:
            return None
        result = []
        for k in keys:
            n = len(key_to_callers[k])
            if n >= min_callers:
                chrom, pos, svtype, svlen = all_repr.get(k, (k[1], k[2] * slop, k[0], 0))
                result.append((chrom, pos, svtype, svlen, n))
        return result or None

    def _run_for(keys, svtype_label, file_prefix):
        counts = collections.Counter(key_to_callers[k] for k in keys)
        mc_variants = _multicaller_variants_for(keys)

        _upset_plot(
            counts,
            caller_names,
            f"Caller comparison — {svtype_label}  (window={slop} bp)",
            os.path.join(output_dir, f"{file_prefix}_upset.jpg"),
            multicaller_variants=mc_variants,
            min_callers=min_callers,
            table_limit=table_limit,
        )

        if mc_variants:
            # Always write TSV
            _write_tsv(
                mc_variants, caller_names,
                os.path.join(output_dir, f"{file_prefix}_multicaller.tsv"),
                min_callers,
            )
            # Genome scatter if too many to fit in table
            if len(mc_variants) > table_limit:
                _genome_scatter_plot(
                    mc_variants,
                    caller_names,
                    f"Multi-caller variants (≥{min_callers} callers) — {svtype_label}",
                    os.path.join(output_dir, f"{file_prefix}_genome_scatter.jpg"),
                )

    # Per-SVTYPE plots
    keys_by_svtype: dict = collections.defaultdict(list)
    for key in universe:
        keys_by_svtype[key[0]].append(key)

    for svtype, keys in sorted(keys_by_svtype.items()):
        _run_for(keys, svtype, f"caller_{svtype}")

    # Combined
    _run_for(list(universe), "all SV types", "caller_combined")
