"""Band extraction, along-diagonal gap-adaptive NaN interpolation, and disk caching.

A *band* is the upper-diagonal representation of a Hi-C contact map: `band[k, j] = C[j-k, j]`,
i.e. row k is the k-th superdiagonal (genomic distance k bins), indexed by the larger
(anchor) coordinate j.  Only `j >= k` is valid; below-diagonal entries are 0.

Missing (unmappable) pixels come back as NaN; `interp_band_along_diagonal` fills them.

Interpolation (Option A, along-diagonal, gap-adaptive): for each missing entry on diagonal k,
`gap` = index-distance to the nearest valid entry on the *same diagonal*; the fill is the mean
of the nearest `round(c * gap)` valid entries on that diagonal (`c` = `neighbor_mult`, default
2).  Sparse regions (large gap) therefore average proportionally more neighbors.
"""
import hashlib
import json
import os

import numpy as np

import cooler
from cooltools.lib.numutils import adaptive_coarsegrain, interp_nan

try:  # jax is optional for pure band I/O
    import jax.numpy as jnp
except Exception:  # pragma: no cover
    jnp = np

# band_cache/ lives one level up (repo root), beside the dlem/ package, not inside it.
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "band_cache")


def _cooler_uri(cool_path: str, resolution: int) -> str:
    """`::resolutions/N` is only valid for multi-resolution (.mcool) containers; a plain
    single-resolution .cool file is opened directly."""
    if cool_path.endswith(".mcool"):
        return f"{cool_path}::resolutions/{resolution}"
    return cool_path


# --------------------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------------------
def extract_band_from_mcool(
    mcool_path: str,
    *,
    resolution: int,
    chrom: str,
    width: int,
    balance: bool = True,
    cool_interp: bool = False,
    coarsegrain: bool = True,
    return_raw: bool = False,
    cutoff: float = 3,
    max_levels: int = 8,
) -> np.ndarray:
    """
    Extract an upper-diagonal band of width `width` from an .mcool file.

    Returns a (width, n_bins) array where row k is the k-th superdiagonal.  Entries below the
    diagonal are zero; unmappable pixels are NaN (unless `cool_interp`).
    """
    if width <= 0:
        raise ValueError("width must be positive.")

    cool = cooler.Cooler(_cooler_uri(mcool_path, resolution))

    bins = cool.bins().fetch(chrom)
    if bins.empty:
        raise ValueError(f"Chromosome {chrom} not found in {mcool_path}.")

    step = width // 2
    block = width + step

    n = len(bins)

    def get_bal(region):
        bal = cool.matrix(balance=True).fetch(region)
        raw = cool.matrix(balance=False).fetch(region)
        # useful for reading custom written coolers
        if return_raw:
            return raw
        if coarsegrain:
            cg = adaptive_coarsegrain(bal, raw, cutoff=cutoff, max_levels=max_levels)
            return interp_nan(cg) if cool_interp else cg
        # No coarse graining; pick balanced or raw, optionally interpolating NaNs.
        base = bal if balance else raw
        return interp_nan(base) if cool_interp else base

    band = np.zeros((width, n), dtype=np.float32)

    for i in range(0, n, step):
        i0, i1 = i, min(n, i + block)
        region = f'{chrom}:{bins.start.iloc[i0]}-{bins.end.iloc[i1-1]}'
        sub = get_bal(region)

        for k in range(width):
            vals = np.diag(sub, k)
            start = i0 + k
            end = start + len(vals)
            if start < n:
                band[k, start:end] = vals[: max(0, n - start)]

    return band


# --------------------------------------------------------------------------------------
# Along-diagonal gap-adaptive interpolation
# --------------------------------------------------------------------------------------
def _fill_1d(v: np.ndarray, c: float) -> np.ndarray:
    """Fill NaNs in a 1D vector by averaging the nearest `round(c*gap)` valid entries.

    gap = index-distance to the nearest valid entry.  The averaged window is taken symmetric
    around the missing entry's insertion point in valid-index space (clipped at the ends),
    which is a vectorized approximation of "the nearest N valid neighbors".
    """
    v = np.asarray(v, dtype=np.float64)
    valid = np.isfinite(v)
    vidx = np.flatnonzero(valid)
    if vidx.size == 0:
        return v  # nothing to interpolate from
    vval = v[vidx]
    prefix = np.concatenate([[0.0], np.cumsum(vval)])  # prefix sums over valid values

    nanp = np.flatnonzero(~valid)
    if nanp.size == 0:
        return v

    ip = np.searchsorted(vidx, nanp)  # insertion point of each NaN in vidx
    big = np.iinfo(np.int64).max
    left_idx = vidx[np.clip(ip - 1, 0, vidx.size - 1)]
    right_idx = vidx[np.clip(ip, 0, vidx.size - 1)]
    dl = np.where(ip > 0, nanp - left_idx, big)
    dr = np.where(ip < vidx.size, right_idx - nanp, big)
    gap = np.minimum(dl, dr)

    N = np.maximum(1, np.round(c * gap).astype(np.int64))
    a = N // 2
    b = N - a
    lo = np.clip(ip - a, 0, vidx.size)
    hi = np.clip(ip + b, 0, vidx.size)
    cnt = hi - lo
    fill = np.where(cnt > 0, (prefix[hi] - prefix[lo]) / np.maximum(cnt, 1), np.nan)

    out = v.copy()
    out[nanp] = fill
    return out


def interp_band_along_diagonal(band: np.ndarray, neighbor_mult: float = 2.0) -> np.ndarray:
    """Fill NaN entries per diagonal using gap-adaptive along-diagonal averaging.

    Args:
        band: (width, n) band with NaN at unmappable pixels.
        neighbor_mult: `c` -- number of valid neighbors averaged = round(c * gap).

    Returns:
        A copy of `band` with NaNs in the valid region (j >= k) filled.
    """
    band = np.asarray(band, dtype=np.float32)
    width, n = band.shape
    out = band.copy()
    for k in range(width):
        if k >= n:
            break
        seg = band[k, k:]  # the diagonal's valid region (anchor j >= k)
        if not np.isnan(seg).any():
            continue
        out[k, k:] = _fill_1d(seg, neighbor_mult).astype(np.float32)
    return out


# --------------------------------------------------------------------------------------
# O/E normalization
# --------------------------------------------------------------------------------------
def normalize_oe(band, log=True):
    """log(observed / expected) per diagonal; expected = mean over valid entries of that diagonal."""
    band = jnp.asarray(band, dtype=jnp.float32)
    if band.ndim != 2:
        raise ValueError("normalize_oe expects a 2D array.")
    width, n = band.shape
    if width == 0 or n == 0:
        return jnp.zeros_like(band)
    row_idx = jnp.arange(width, dtype=jnp.int32)[:, None]
    col_idx = jnp.arange(n, dtype=jnp.int32)[None, :]
    mask = col_idx >= row_idx
    masked_band = jnp.where(mask, band, 0.0)
    counts = mask.sum(axis=1).astype(band.dtype)
    sums = masked_band.sum(axis=1)
    means = jnp.where(counts > 0, sums / counts, 0.0)
    means_expanded = means[:, None]
    offset = 1e-6
    valid = mask & (means_expanded > 0.0)
    denom = jnp.where(valid, means_expanded, 1.0)
    ratios = jnp.where(valid, (band + offset) / (denom + offset), 1.0)
    if not log:
        return jnp.where(valid, ratios, 0.0)
    return jnp.where(valid, jnp.log(ratios), 0.0)


# --------------------------------------------------------------------------------------
# Cached loader
# --------------------------------------------------------------------------------------
def _cache_key(mcool_path, resolution, chrom, width, neighbor_mult, fetch_kwargs) -> str:
    st = os.stat(mcool_path)
    payload = {
        "mcool": os.path.abspath(mcool_path),
        "size": st.st_size,
        "mtime": int(st.st_mtime),
        "resolution": resolution,
        "chrom": chrom,
        "width": width,
        "neighbor_mult": neighbor_mult,
        "fetch": {k: fetch_kwargs[k] for k in sorted(fetch_kwargs)},
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def load_band(
    mcool_path: str,
    *,
    resolution: int,
    chrom: str,
    width: int,
    neighbor_mult: float = 2.0,
    cache_dir: str = CACHE_DIR,
    use_cache: bool = True,
    verbose: bool = True,
    **fetch_kwargs,
) -> np.ndarray:
    """Fetch a band, interpolate NaNs along the diagonal, and cache to disk.

    Cache key hashes (mcool path/size/mtime, resolution, chrom, width, neighbor_mult, fetch kwargs).
    Extra kwargs (balance, coarsegrain, cutoff, ...) are forwarded to `extract_band_from_mcool`.
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = _cache_key(mcool_path, resolution, chrom, width, neighbor_mult, fetch_kwargs)
    path = os.path.join(cache_dir, f"band_{chrom}_{resolution}_{width}_{key}.npy")
    if use_cache and os.path.exists(path):
        if verbose:
            print(f"[band] cache hit {path}")
        return np.load(path)
    if verbose:
        print(f"[band] fetching {chrom} @ {resolution} width={width} ...", flush=True)
    raw = extract_band_from_mcool(
        mcool_path, resolution=resolution, chrom=chrom, width=width, **fetch_kwargs
    )
    band = interp_band_along_diagonal(raw, neighbor_mult=neighbor_mult)
    if use_cache:
        np.save(path, band)
        if verbose:
            print(f"[band] cached -> {path}")
    return band


# --------------------------------------------------------------------------------------
# Dense patch (for eyeballing a region / probing balancing & coarsegrain)
# --------------------------------------------------------------------------------------
def load_patch(mcool_path, *, resolution, chrom, start, span,
               balance=True, coarsegrain=True, cutoff=3, max_levels=8, cool_interp=False):
    """Load a dense (n x n) contact matrix for [start, start+span) bp of `chrom` (uncached).

    Exposes the balancing / coarse-grain knobs so you can see how they change a region.
    Returns the dense matrix (NaN at unmappable pixels unless cool_interp).
    """
    cool = cooler.Cooler(_cooler_uri(mcool_path, resolution))
    region = f"{chrom}:{int(start)}-{int(start + span)}"
    bal = cool.matrix(balance=True).fetch(region)
    raw = cool.matrix(balance=False).fetch(region)
    if coarsegrain:
        cg = adaptive_coarsegrain(bal, raw, cutoff=cutoff, max_levels=max_levels)
        return interp_nan(cg) if cool_interp else cg
    base = bal if balance else raw
    return interp_nan(base) if cool_interp else base


# --------------------------------------------------------------------------------------
# CTCF (or any +/- stranded pair) from a per-bin track .npz  (data/tracks/<cell>_<res>_hg38.npz)
# --------------------------------------------------------------------------------------
def load_ctcf(track_npz, chrom, *, resolution, n_bins, pos="CTCF+", neg="CTCF-"):
    """Return (ctcf_pos, ctcf_neg) per-bin vectors (length n_bins) for `chrom` from a track npz.

    The npz has arrays track_label, chrom, bin_start, mean[n_bins, n_tracks]. Edit this loader
    for your own track format.
    """
    import sys as _sys, importlib.util as _ilu
    if _ilu.find_spec("numpy._core") is None:      # npz written with numpy>=2
        import numpy.core as _c
        for _n in list(_sys.modules):
            if _n == "numpy.core" or _n.startswith("numpy.core."):
                _sys.modules[_n.replace("numpy.core", "numpy._core", 1)] = _sys.modules[_n]
        _sys.modules.setdefault("numpy._core", _c)
    d = np.load(track_npz, allow_pickle=True)
    labels = [str(x) for x in d["track_label"]]
    m = (d["chrom"] == chrom)
    binpos = (d["bin_start"][m] // resolution).astype(int)

    def _track(suffix):
        idx = [i for i, lab in enumerate(labels) if lab.endswith(suffix)]
        vals = d["mean"][m][:, idx].mean(axis=1).astype(np.float32)
        out = np.zeros(n_bins, np.float32)
        keep = binpos < n_bins
        out[binpos[keep]] = vals[keep]
        return out

    return _track(pos), _track(neg)
