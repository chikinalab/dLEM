"""Data-vs-prediction contact-map plots (numpy + matplotlib only; no GPU).

Bands are square double-triangles: upper triangle = prediction, lower triangle = data, on a
diverging (vlag) log-O/E scale, with optional CTCF / L-R barrier tracks underneath.
"""
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .io import normalize_oe


def _resolve_cmap(name):
    """Return `name` if matplotlib knows it, else register seaborn's version, else RdBu_r."""
    try:
        plt.get_cmap(name)
        return name
    except Exception:
        pass
    try:
        import seaborn as sns

        return sns.color_palette(name, as_cmap=True)
    except Exception:
        return "RdBu_r"


def band_to_dense(band, i0, i1):
    """Reconstruct a symmetric (w, w) contact matrix for window [i0, i1) from a band.

    band[k, j] = C[j-k, j]; here M[p, q] = band[q-p, i0+q] for q >= p (upper), symmetric.
    """
    band = np.asarray(band)
    w = i1 - i0
    width = band.shape[0]
    M = np.zeros((w, w), dtype=np.float32)
    for p in range(w):
        for q in range(p, w):
            k = q - p
            if k < width and (i0 + q) < band.shape[1]:
                M[p, q] = M[q, p] = band[k, i0 + q]
    return M


def _double_triangle(ax, upper_band, lower_band, i0, i1, *, cmap="vlag", vmax=None):
    U = band_to_dense(upper_band, i0, i1)
    Lo = band_to_dense(lower_band, i0, i1)
    w = i1 - i0
    tri = np.triu_indices(w)
    M = np.zeros((w, w), dtype=np.float32)
    M[tri] = U[tri]
    M[(tri[1], tri[0])] = Lo[(tri[1], tri[0])]
    if vmax is None:
        vmax = np.nanpercentile(np.abs(M), 99) or 1.0
    ax.imshow(M, cmap=_resolve_cmap(cmap), vmin=-vmax, vmax=vmax, interpolation="none")
    ax.set_xticks([])
    ax.set_yticks([])
    return vmax


def plot_prediction(
    data_oe,
    pred_oe,
    window,
    *,
    tracks=None,
    barrier=None,
    cmap="vlag",
    vmax=None,
    title=None,
    out=None,
):
    """Double-triangle plot (upper=pred, lower=data) for a genomic window.

    Args:
        data_oe: log-O/E band (width, n), shown in the lower triangle.
        pred_oe: log-O/E band (width, n), shown in the upper triangle.
        window: (i0, i1) bin range.
        tracks: optional {name: 1D array over full n} plotted as a track under the map,
                sliced to the window.
        barrier: optional 1D array (e.g. (1-L)(1-R)) plotted as a barrier track.
        out: if given, save the figure there; otherwise return (fig, axes).
    """
    i0, i1 = window
    ntrk = (len(tracks) if tracks else 0) + (1 if barrier is not None else 0)
    fig, axes = plt.subplots(
        1 + ntrk, 1, figsize=(5, 5 + 0.6 * ntrk),
        gridspec_kw={"height_ratios": [5] + [0.6] * ntrk}, squeeze=False,
    )
    axes = axes[:, 0]
    vmax = _double_triangle(axes[0], pred_oe, data_oe, i0, i1, cmap=cmap, vmax=vmax)
    axes[0].set_title(title or "upper = prediction, lower = data")
    ai = 1
    xr = np.arange(i0, i1)
    if barrier is not None:
        axes[ai].plot(np.arange(i1 - i0), np.asarray(barrier)[i0:i1], lw=0.8, color="k")
        axes[ai].set_ylabel("(1-L)(1-R)", fontsize=7, rotation=0, ha="right", va="center")
        axes[ai].set_xlim(0, i1 - i0 - 1); axes[ai].set_yticks([]); axes[ai].set_xticks([])
        ai += 1
    for name, vec in (tracks or {}).items():
        axes[ai].plot(np.arange(i1 - i0), np.asarray(vec)[i0:i1], lw=0.8, color="C3")
        axes[ai].set_ylabel(name, fontsize=7, rotation=0, ha="right", va="center")
        axes[ai].set_xlim(0, i1 - i0 - 1); axes[ai].set_yticks([]); axes[ai].set_xticks([])
        ai += 1
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out
    return fig, axes


# --------------------------------------------------------------------------------------
# Dense patch + fit-region convenience plots
# --------------------------------------------------------------------------------------
def plot_patch(patch, *, scale="log", ax=None, interpolation="nearest", title=None, cmap=None):
    """Show a dense contact patch. scale="log" -> log1p; scale="detrend" -> per-diagonal log-O/E."""
    patch = np.asarray(patch, dtype=np.float64)
    if scale == "log":
        img = np.log1p(np.nan_to_num(patch)); cmap = cmap or "magma"; vmin = vmax = None
    elif scale == "detrend":
        # O/E is per-diagonal (genomic distance): compute the expected on the band axis
        # (one mean per diagonal), then expand onto the dense pixel grid as the last step.
        w = patch.shape[0]
        exp = np.array([np.nanmean(np.diagonal(patch, k)) for k in range(w)])   # expected per distance
        dist = np.abs(np.subtract.outer(np.arange(w), np.arange(w)))            # |p-q| per pixel
        mu = exp[dist]                                                          # expand to dense (last step)
        oe = np.where(np.isfinite(patch) & (mu > 0), np.log((patch + 1e-6) / (mu + 1e-6)), 0.0)
        img = oe; cmap = cmap or _resolve_cmap("vlag")
        v = np.nanpercentile(np.abs(img), 99) or 1.0; vmin, vmax = -v, v
    else:
        raise ValueError("scale must be 'log' or 'detrend'")
    if ax is None:
        import matplotlib.pyplot as _plt
        _, ax = _plt.subplots(figsize=(4, 4))
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interpolation)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def plot_prediction_region(fit, band, *, start, span, extent=None, checkpoint=None,
                           top_tracks=None, barrier=True, cmap="vlag", vmax=None, title=None, out=None):
    """Double-triangle prediction-vs-data plot for a window. All of start / span / extent are in
    BIN units (not bp).

    start, span  -- the window is columns [start, start+span) of the band (span bins wide).
    extent       -- how many diagonals (genomic-distance rows) to render; default = span (the full
                    triangle), capped at span (a span-wide window can't show more). The PREDICTION
                    fills all `extent` diagonals; the DATA only has band.shape[0] diagonals, so if
                    extent > band depth the data (lower) triangle is filled to the band depth and
                    empty beyond (an honest bowtie). For two exactly-matching halves, use
                    extent <= band.shape[0].
    checkpoint   -- which checkpoint of the DlemFit to plot ("mse","corr","loss","symm","symm2");
                    default None uses the fit's selected monitor (fit.L/fit.R).

    The prediction (upper) and data (lower) triangles both show `extent` diagonals. The prediction
    comes from `fit.predict(extent, checkpoint=checkpoint)` -- so this works with ANY fit object that
    implements `.predict(rows, checkpoint=None)` and `.barrier(checkpoint=None)`, not just the base
    L/R DlemFit.
    """
    if not (hasattr(fit, "predict") and hasattr(fit, "barrier")):
        raise TypeError("plot_prediction_region needs a fit object with .predict(rows, checkpoint=) "
                        "and .barrier(checkpoint=) methods (e.g. a DlemFit).")
    extent = min(int(extent or span), span)   # a span-wide window shows at most span diagonals
    name = checkpoint or getattr(fit, "monitor", "fit")
    data_oe = np.asarray(normalize_oe(band[:extent]))                    # lower triangle
    pred_oe = np.asarray(normalize_oe(fit.predict(extent, checkpoint=checkpoint)))  # upper triangle
    bar = np.asarray(fit.barrier(checkpoint=checkpoint)) if barrier else None
    tracks = {"CTCF": np.asarray(top_tracks)} if top_tracks is not None else None
    ttl = title or (f"{name}: upper=pred, lower=data (log-O/E)  "
                    f"bins {start}-{start + span}, extent {extent}")
    return plot_prediction(data_oe, pred_oe, (start, start + span), tracks=tracks, barrier=bar,
                           cmap=cmap, vmax=vmax, title=ttl, out=out)
