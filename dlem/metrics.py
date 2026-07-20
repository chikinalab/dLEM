"""Lagged correlation of L/R fields with genomic tracks (e.g. CTCF).
"""
import numpy as np


def max_lagged_correlation(a, b, *, max_lag=10):
    """Max Pearson correlation between a and b allowing shifts up to +/-max_lag.

    Returns (best_corr, best_lag); best_lag is the shift applied to b (positive = b right).
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Inputs must be 1D vectors.")
    if len(a) != len(b):
        length = min(len(a), len(b))
        a = a[:length]
        b = b[:length]

    best_corr = 0.0
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            a_slice, b_slice = a[lag:], b[:-lag]
        elif lag < 0:
            a_slice, b_slice = a[:lag], b[-lag:]
        else:
            a_slice, b_slice = a, b
        if len(a_slice) < 2:
            continue
        corr = np.corrcoef(a_slice, b_slice)[0, 1]
        if np.isfinite(corr) and abs(corr) > abs(best_corr):
            best_corr, best_lag = corr, lag
    return best_corr, best_lag


def load_ctcf_tsv(comp_tsv, chrom):
    """Load the painted CTCF tracks for `chrom` from a comp table.

    Returns (cpos, cneg), each length == n_bins for the chromosome (columns
    CTCF_painted_pos / CTCF_painted_neg).  Raises if the chromosome is absent.
    """
    import pandas as pd

    t = pd.read_csv(comp_tsv, sep="\t")
    cc = t.loc[t["chrID"] == chrom]
    if cc.empty:
        raise ValueError(f"{chrom} not in comp table {comp_tsv}")
    return (cc["CTCF_painted_pos"].to_numpy(np.float32),
            cc["CTCF_painted_neg"].to_numpy(np.float32))


def ctcf_alignment(L, R, cpos, cneg, *, max_lag=10):
    """Max |lagged corr| over {cpos, cneg} x {L, R} -- the standard CTCF alignment score.

    The dLEM alignment is an anti-correlation (CTCF peak <-> L/R barrier dip), so the score is
    the absolute lagged correlation.  Vectors are length-matched to the shorter of band/track.

    Returns (score, best_field, best_track, best_lag).
    """
    L = np.asarray(L, np.float32)
    R = np.asarray(R, np.float32)
    best = (0.0, None, None, 0)
    for tname, t in (("pos", cpos), ("neg", cneg)):
        for fname, f in (("L", L), ("R", R)):
            m = min(len(f), len(t))
            c, lag = max_lagged_correlation(f[:m], np.asarray(t)[:m], max_lag=max_lag)
            if abs(c) > abs(best[0]):
                best = (float(c), fname, tname, int(lag))
    return abs(best[0]), best[1], best[2], best[3]


def track_correlations(fields: dict, track, *, max_lag=10):
    """Max |lagged corr| of each named field vs a track.

    fields: {name: 1D array}, e.g. {"L": L, "R": R}.
    Returns {name: (abs_corr, signed_corr, lag)}.
    """
    track = np.asarray(track, dtype=np.float32)
    out = {}
    for name, vec in fields.items():
        vec = np.asarray(vec, dtype=np.float32)
        c, lag = max_lagged_correlation(vec, track, max_lag=max_lag)
        out[name] = (abs(c), float(c), int(lag))
    return out


# --------------------------------------------------------------------------------------
# High-level reporting
# --------------------------------------------------------------------------------------
def lagged_corr(a, b, *, max_lag=10):
    """Best |lagged Pearson corr| between a and b over shifts +/- max_lag. Returns (corr, lag)."""
    return max_lagged_correlation(a, b, max_lag=max_lag)


def ctcf_report(fit, ctcf_pos, ctcf_neg, *, checkpoint="symm2", max_lag=2, verbose=True):
    """CTCF+/- correlation of the L/R barrier tracks (barriers are dips -> anti-correlation
    expected, split by motif strand: L~CTCF+, R~CTCF-. Previously stated as R~CTCF+/L~CTCF-
    under the old, swapped L/R convention -- see dlem/model.py's module docstring; this is
    the same empirical alignment under the corrected names, not a new finding).

    checkpoint: which checkpoint of a DlemFit to report -- "symm2" (DEFAULT), "symm", "mse",
        "corr", "loss", or "all" for every stored checkpoint (one row each; the fit's selected
        `monitor` is marked '*'). Falls back to the fit's own (L, R) if it carries no such
        checkpoint, or if `fit` is a bare (L, R) tuple.
    Returns {checkpoint_name: {"L.CTCF+","L.CTCF-","R.CTCF+","R.CTCF-"}}.
    """
    ckpts = getattr(fit, "checkpoints", None)
    selected = getattr(fit, "monitor", None)
    if ckpts and checkpoint == "all":
        items = list(ckpts.items())
    elif ckpts and checkpoint in ckpts:
        items = [(checkpoint, ckpts[checkpoint])]
    elif hasattr(fit, "L"):                      # DlemFit without that checkpoint, or none stored
        items = [(selected or "fit", (fit.L, fit.R))]
    else:                                         # bare (L, R) tuple
        items = [("fit", fit)]

    sc = lambda x, y: max_lagged_correlation(np.clip(np.asarray(x), 1e-4, 1.0),
                                             np.asarray(y), max_lag=max_lag)[0]
    out = {}
    if verbose:
        print(f"CTCF correlation (anti-correlation expected; L~CTCF+, R~CTCF-), max_lag={max_lag}:")
        print(f"  {'checkpoint':12s} {'L.CTCF+':>8s} {'L.CTCF-':>8s} {'R.CTCF+':>8s} {'R.CTCF-':>8s}")
    for name, (L, R) in items:
        r = {"L.CTCF+": sc(L, ctcf_pos), "L.CTCF-": sc(L, ctcf_neg),
             "R.CTCF+": sc(R, ctcf_pos), "R.CTCF-": sc(R, ctcf_neg)}
        out[name] = r
        if verbose:
            mark = " *" if name == selected else ""
            print(f"  {name:12s} {r['L.CTCF+']:+8.3f} {r['L.CTCF-']:+8.3f} "
                  f"{r['R.CTCF+']:+8.3f} {r['R.CTCF-']:+8.3f}{mark}")
    return out
