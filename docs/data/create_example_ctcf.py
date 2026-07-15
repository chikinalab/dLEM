"""
Creates docs/data/example_ctcf.tsv: small CTCF tracks for the same `ref_region`
window as docs/data/example_chr10.cool (chr10:19-26 Mb, 700 bins at 10 kb --
local bins 0-699 = chr10 bins 1900-2599).

Run once (requires local access to Maria's H1-hESC comp table):
    python docs/data/create_example_ctcf.py

Source : H1-hESC CTCF tracks, ~/MariasLoopCode/loopOptimize/H1hESC_comp_table_all.tsv
         (columns: coordinate [0-based per-chromosome 10kb bin], chrID, CTCF
         [plain ChIP signal, not strand-split], CTCF_painted_pos, CTCF_painted_neg
         [motif-strand-oriented]).
Format : chrID, CTCF, CTCF_painted_pos, CTCF_painted_neg -- matches dlem.load_ctcf_tsv
         (which reads the painted pair) plus the plain CTCF column for combined-signal
         use (e.g. against (1-L)+(1-R)/(1-L)(1-R), which don't have a strand to split).
         chrID relabeled 'ref_region' to match example_chr10.cool.
"""
import os
import pandas as pd

COMP_TSV = os.path.expanduser(
    "~/MariasLoopCode/loopOptimize/H1hESC_comp_table_all.tsv"
)
CHROM = "chr10"
BIN_START, BIN_END = 1900, 2600  # chr10:19-26 Mb at 10kb -- matches example_chr10.cool exactly
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_ctcf.tsv")

t = pd.read_csv(COMP_TSV, sep="\t",
                usecols=["coordinate", "chrID", "CTCF", "CTCF_painted_pos", "CTCF_painted_neg"])
c = t[t["chrID"] == CHROM].sort_values("coordinate").reset_index(drop=True)
sl = c.iloc[BIN_START:BIN_END].copy()
assert len(sl) == BIN_END - BIN_START, f"expected {BIN_END - BIN_START} rows, got {len(sl)}"
sl["chrID"] = "ref_region"
sl = sl[["chrID", "CTCF", "CTCF_painted_pos", "CTCF_painted_neg"]]

sl.to_csv(OUT, sep="\t", index=False)
print(f"Written: {OUT}  ({len(sl)} bins, chr10:{BIN_START*10_000}-{BIN_END*10_000})")
