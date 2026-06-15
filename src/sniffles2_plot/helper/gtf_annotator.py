# -*- coding: utf-8 -*-
"""
Lightweight GTF annotation helper.

Parses gene-level records from a GTF/GTF.gz file and provides fast
overlap / nearest-gene queries using per-chromosome sorted interval lists
and bisect — no external dependencies beyond the standard library.
"""
import bisect
import gzip
import re
from collections import defaultdict


def _open_gtf(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


_GENE_NAME_RE = re.compile(r'gene_name\s+"([^"]+)"')
_GENE_ID_RE   = re.compile(r'gene_id\s+"([^"]+)"')


def _extract_gene_name(attributes: str) -> str:
    m = _GENE_NAME_RE.search(attributes)
    if m:
        return m.group(1)
    m = _GENE_ID_RE.search(attributes)
    return m.group(1) if m else "."


class GeneAnnotator:
    """
    Provides annotate(chrom, start, end) and nearest(chrom, pos, max_dist)
    queries against a GTF gene index.
    """

    def __init__(self, gtf_path: str):
        # Per-chromosome: list of (start, end, gene_name) sorted by start
        self._index: dict = defaultdict(list)
        self._load(gtf_path)

    def _load(self, gtf_path: str):
        print(f"Loading GTF: {gtf_path}")
        n = 0
        with _open_gtf(gtf_path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 9:
                    continue
                feature = parts[2]
                if feature != "gene":
                    continue
                chrom = parts[0]
                try:
                    start = int(parts[3])
                    end   = int(parts[4])
                except ValueError:
                    continue
                gene_name = _extract_gene_name(parts[8])
                self._index[chrom].append((start, end, gene_name))
                n += 1

        # Sort by start and build parallel start-only list for bisect
        self._starts: dict = {}
        for chrom, genes in self._index.items():
            genes.sort()
            self._starts[chrom] = [g[0] for g in genes]

        print(f"  Loaded {n} gene records across {len(self._index)} chromosomes.")

    def annotate(self, chrom: str, start: int, end: int) -> list:
        """Return gene names overlapping [start, end] on chrom."""
        genes = self._index.get(chrom) or self._index.get(_strip_chr(chrom))
        starts = self._starts.get(chrom) or self._starts.get(_strip_chr(chrom))
        if not genes:
            return []
        # All genes whose start <= end
        idx = bisect.bisect_right(starts, end)
        result = []
        for i in range(idx - 1, -1, -1):
            g_start, g_end, g_name = genes[i]
            if g_end < start:
                break
            result.append(g_name)
        return result

    def nearest(self, chrom: str, pos: int, max_dist: int = 100_000) -> str:
        """Return the name of the nearest gene within max_dist of pos."""
        genes = self._index.get(chrom) or self._index.get(_strip_chr(chrom))
        starts = self._starts.get(chrom) or self._starts.get(_strip_chr(chrom))
        if not genes:
            return "."
        # Check overlap first
        overlapping = self.annotate(chrom, pos, pos)
        if overlapping:
            return overlapping[0]
        # Find nearest by distance to gene body
        idx = bisect.bisect_left(starts, pos)
        best_name, best_dist = ".", max_dist + 1
        for i in [idx - 1, idx]:
            if 0 <= i < len(genes):
                g_start, g_end, g_name = genes[i]
                dist = max(0, g_start - pos, pos - g_end)
                if dist < best_dist:
                    best_dist, best_name = dist, g_name
        return best_name if best_dist <= max_dist else "."


def _strip_chr(chrom: str) -> str:
    """Return chrom with 'chr' prefix stripped, for cross-reference matching."""
    return chrom[3:] if chrom.startswith("chr") else chrom
