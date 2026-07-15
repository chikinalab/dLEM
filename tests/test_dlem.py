import os

import numpy as np
import pandas as pd
import cooler
import pytest
from typer.testing import CliRunner

import dlem
from dlem.cli import dlem_cli

runner = CliRunner()


# --------------------------------------------------------------------------------------
# Real bundled data (docs/data/example_chr10.cool + example_ctcf.tsv): fast, network-free,
# exercises the actual fit -> CTCF-alignment pipeline against real barrier/CTCF structure.
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_fit(example_cool):
    band = dlem.load_band(example_cool, resolution=10_000, chrom="ref_region", width=700)
    fit = dlem.fit(band, s=0.025, train_to=200, importance_power=0.5,
                   diag_weight_power=1.0, n_opt=500, verbose=False)
    return band, fit


def test_fit_real_data(real_fit):
    _, fit = real_fit
    assert np.isfinite(fit.mse) and np.isfinite(fit.corr)
    assert fit.mse < 0.5
    assert fit.corr > 0.4


def test_ctcf_alignment_real_data(real_fit, example_ctcf_tsv):
    _, fit = real_fit
    cpos, cneg = dlem.load_ctcf_tsv(example_ctcf_tsv, "ref_region")
    report = dlem.ctcf_report(fit, cpos, cneg, max_lag=2, verbose=False)
    vals = report[fit.monitor]
    # dLEM barriers are dips -> real CTCF alignment is a genuine anti-correlation, not noise.
    assert all(v < 0 for v in vals.values())
    assert max(abs(v) for v in vals.values()) > 0.15


# --------------------------------------------------------------------------------------
# Synthetic data: pure mechanism check (fit runs and produces finite output), independent
# of any bundled data file.
# --------------------------------------------------------------------------------------
def synthetic_cool(path, *, resolution=1000, chr_size=30_000, ct_mean=5.0, ct_sd=1.0,
                   diag_vs_rest=2.0, seed=24):
    rng = np.random.default_rng(seed)
    chromsizes = pd.Series({"chr1": chr_size})
    bins = cooler.binnify(chromsizes, resolution)
    bins["weight"] = 1.0
    n_bins = len(bins)

    bin1_ids = np.arange(n_bins)
    bin2_ids = np.arange(n_bins)
    counts = rng.normal(ct_mean * diag_vs_rest, ct_sd, size=n_bins)

    off_diag_bin1 = rng.integers(0, n_bins, size=n_bins)
    off_diag_bin2 = rng.integers(0, n_bins, size=n_bins)
    nondiag = off_diag_bin1 != off_diag_bin2
    off_diag_counts = rng.normal(ct_mean, ct_sd, size=n_bins)

    bin1_ids = np.concatenate([bin1_ids, off_diag_bin1[nondiag]])
    bin2_ids = np.concatenate([bin2_ids, off_diag_bin2[nondiag]])
    counts = np.concatenate([counts, off_diag_counts[nondiag]])

    mask = bin1_ids <= bin2_ids
    pixels = pd.DataFrame({
        "bin1_id": bin1_ids[mask], "bin2_id": bin2_ids[mask], "count": counts[mask],
    }).sort_values(["bin1_id", "bin2_id"]).drop_duplicates(
        subset=["bin1_id", "bin2_id"]).reset_index(drop=True)

    cooler.create_cooler(path, bins, pixels, assembly="synthetic_assembly")


def test_fit_synthetic_smoke(tmp_path):
    cool_path = str(tmp_path / "synthetic.cool")
    synthetic_cool(cool_path)
    band = dlem.load_band(cool_path, resolution=1000, chrom="chr1", width=20, use_cache=False)
    fit = dlem.fit(band, s=0.025, train_to=20, n_opt=50, verbose=False)
    assert np.isfinite(fit.mse) and np.isfinite(fit.corr)
    assert np.all(np.isfinite(fit.L)) and np.all(np.isfinite(fit.R))


def test_fit_lean_smoke(tmp_path):
    """fit_lean (the lower-level (L, R)-only entry point) gets no other coverage otherwise."""
    cool_path = str(tmp_path / "synthetic.cool")
    synthetic_cool(cool_path)
    band = dlem.load_band(cool_path, resolution=1000, chrom="chr1", width=20, use_cache=False)
    L, R = dlem.fit_lean(band, s=0.025, train_to=20, n_opt=50, verbose=False)
    assert np.all(np.isfinite(L)) and np.all(np.isfinite(R))


# --------------------------------------------------------------------------------------
# CLI: confirm the console script itself still works end-to-end.
# --------------------------------------------------------------------------------------
def test_cli_synthetic_end_to_end(tmp_path):
    cool_path = str(tmp_path / "synthetic.cool")
    synthetic_cool(cool_path)
    out_dir = str(tmp_path / "out")
    result = runner.invoke(dlem_cli, [
        cool_path, out_dir,
        "--chrom", "chr1", "--resolution", "1000", "--width", "20", "--train-to", "20",
        "--n-opt", "50", "--plot", "--plot-span", "20",
    ])
    assert result.exit_code == 0, result.output
    assert os.path.exists(os.path.join(out_dir, "dlem_fit.pkl"))
    assert os.path.exists(os.path.join(out_dir, "prediction.png"))
