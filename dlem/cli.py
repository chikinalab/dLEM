"""dLEM command line interface.

A thin wrapper around the library's single-region workflow (matching what
docs/quick_start.ipynb does): load_band -> fit -> optional ctcf_report ->
optional plot_prediction_region.

Scope note: this does not (yet) reproduce the old CLI's genome-wide batch
looping (multi-chromosome / bed-region), bigWig track output, or predicted
.cool output -- none of those have an equivalent in the ported engine. Use
the `dlem` library directly (see docs/quick_start.ipynb) for anything beyond
a single chromosome/region.
"""
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich import print

from dlem.io import load_band
from dlem.train import fit as dlem_fit
from dlem.metrics import ctcf_report, load_ctcf_tsv
from dlem.viz import plot_prediction_region

dlem_cli = typer.Typer(
    help="""Differentiable Loop Extrusion Model (dLEM): fit L/R extrusion-barrier tracks
    to a single chromosome/region of Hi-C or Micro-C data and report/plot the result.
    Implemented in Jax; runs on CPU or GPU (select via JAX_PLATFORMS before launch).""",
    context_settings={"help_option_names": ["-h", "--help"]},
)


@dlem_cli.command()
def main(
    dna_ints: Path = typer.Argument(
        ..., help="Path to a .cool file, or a .mcool file (use --resolution)."
    ),
    output_location: Path = typer.Argument(
        ..., help="Output directory (created if missing)."
    ),
    chrom: str = typer.Option(
        ..., "--chrom", "-c", help="Chromosome to load (dlem.load_band's `chrom`)."
    ),
    resolution: int = typer.Option(
        ..., "--resolution", "-res", help="Bin resolution in bp."
    ),
    width: int = typer.Option(
        200, "--width", "-w", help="Band depth (diagonals) to load."
    ),
    train_to: Optional[int] = typer.Option(
        None, "--train-to", help="Diagonals to fit (default: --width)."
    ),
    slowdown: float = typer.Option(
        0.025, "--slowdown", "-s", help="dLEM slowdown constant (per-bin)."
    ),
    importance_power: float = typer.Option(
        0.0, "--importance-power", help="Within-diagonal emphasis on high-mass bins."
    ),
    diag_weight_power: float = typer.Option(
        1.0, "--diag-weight-power", help="Between-diagonal weighting by mass^power."
    ),
    n_opt: int = typer.Option(1500, "--n-opt", help="Adam optimization steps."),
    monitor: str = typer.Option(
        "symm2", "--monitor", help="Checkpoint to select: mse|corr|loss|symm|symm2."
    ),
    ctcf_tsv: Optional[Path] = typer.Option(
        None, "--ctcf-tsv",
        help="Comp-table TSV (dlem.load_ctcf_tsv format) to report CTCF alignment against.",
    ),
    plot: bool = typer.Option(
        False, "--plot", help="Save a data-vs-prediction PNG (prediction.png)."
    ),
    plot_start: int = typer.Option(0, "--plot-start", help="Plot window start (bins)."),
    plot_span: Optional[int] = typer.Option(
        None, "--plot-span", help="Plot window span (bins); default: --train-to."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Print optimization progress."),
):
    """Fit dLEM to one chromosome/region and report mse/corr (+ optional CTCF/plot)."""
    train_to = train_to or width
    output_location = output_location.resolve()
    os.makedirs(output_location, exist_ok=True)

    print(f"Loading {chrom} @ {resolution}bp, width={width} from {dna_ints}")
    band = load_band(str(dna_ints), resolution=resolution, chrom=chrom, width=width)

    print(f"Fitting dLEM (train_to={train_to}, monitor={monitor}, n_opt={n_opt})")
    result = dlem_fit(
        band, s=slowdown, train_to=train_to, importance_power=importance_power,
        diag_weight_power=diag_weight_power, monitor=monitor, n_opt=n_opt, verbose=verbose,
    )
    print(f"[dlem] mse={result.mse:.4f}  corr={result.corr:.4f}  checkpoint={result.monitor}")

    out_pickle = output_location / "dlem_fit.pkl"
    with open(out_pickle, "wb") as fh:
        pickle.dump(result, fh)
    print(f"Saved fit to {out_pickle}")

    if ctcf_tsv is not None:
        cpos, cneg = load_ctcf_tsv(str(ctcf_tsv), chrom)
        ctcf_report(result, cpos, cneg)

    if plot:
        span = plot_span or train_to
        out_png = output_location / "prediction.png"
        plot_prediction_region(result, band, start=plot_start, span=span, out=str(out_png))
        print(f"Saved plot to {out_png}")
