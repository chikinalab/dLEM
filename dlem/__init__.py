"""dlem -- Differentiable Loop Extrusion Model (dLEM) of Hi-C contact maps:
fitting, metrics, and visualization behind a single flat import.

    import dlem
    band = dlem.load_band("docs/data/example_chr10.cool", resolution=10000, chrom="ref_region", width=700)
    fit  = dlem.fit(band, s=0.025, train_to=200)                        # -> DlemFit
    cpos, cneg = dlem.load_ctcf_tsv("docs/data/example_ctcf.tsv", "ref_region")
    dlem.ctcf_report(fit, cpos, cneg)                                  # checkpoint="all" to compare
    dlem.plot_prediction_region(fit, band, start=150, span=200, top_tracks=cpos)

Public API
  I/O     : load_band, load_patch, load_ctcf, load_ctcf_tsv, normalize_oe
  fit     : fit -> DlemFit  (also fit_lean, run_adam)
  model   : full_rollout, full_rollout_selfnorm
  metrics : lagged_corr, ctcf_report, max_lagged_correlation, track_correlations
  viz     : plot_patch, plot_prediction, plot_prediction_region

See docs/quick_start.ipynb for an end-to-end walkthrough. Internals live in submodules
(io, model, losses, train, metrics, viz).
"""
from .io import load_band, load_patch, load_ctcf, normalize_oe, extract_band_from_mcool
from .model import full_rollout, full_rollout_selfnorm
from .losses import make_losses, band_metrics
from .train import fit, DlemFit, fit_lean, run_adam
from .metrics import (lagged_corr, ctcf_report, max_lagged_correlation,
                      track_correlations, load_ctcf_tsv)
from .viz import (plot_patch, plot_prediction, plot_prediction_region,
                  band_to_dense)

__all__ = [
    "load_band", "load_patch", "load_ctcf", "load_ctcf_tsv", "normalize_oe",
    "extract_band_from_mcool",
    "fit", "DlemFit", "fit_lean", "run_adam",
    "full_rollout", "full_rollout_selfnorm",
    "make_losses", "band_metrics",
    "lagged_corr", "ctcf_report", "max_lagged_correlation", "track_correlations",
    "plot_patch", "plot_prediction", "plot_prediction_region", "band_to_dense",
]

__version__ = "0.1.0"


def _check_gpu():
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi'], capture_output=True, timeout=3
        )
        if result.returncode != 0:
            return  # no NVIDIA GPU
        import jax
        if jax.default_backend() == 'cpu':
            print(
                '[dlem] NVIDIA GPU detected but JAX is running on CPU.\n'
                '       For GPU acceleration reinstall with:\n'
                '           pip install "dlem-jax[cuda]"\n',
                flush=True,
            )
    except Exception:
        pass  # never crash on import due to this check


_check_gpu()
