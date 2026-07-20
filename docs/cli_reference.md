# `dLEM`

dLEM: differentiable Loop Extrusion Model. Fits `L`/`R` extrusion-barrier tracks to a
single chromosome/region and reports mse/corr (+ optional CTCF alignment / plot).

**Usage**:

```console
$ dlem [OPTIONS] DNA_INTS OUTPUT_LOCATION
```

**Arguments**:

* `DNA_INTS`: Path to a .cool file, or a .mcool file (use `--resolution`)  [required]
* `OUTPUT_LOCATION`: Output directory, will be created if not existing  [required]

**Options**:

* `-c, --chrom TEXT`: Chromosome to load (`dlem.load_band`'s `chrom`)  [required]
* `-res, --resolution INTEGER`: Bin resolution in bp  [required]
* `-w, --width INTEGER`: Band depth (diagonals) to load  [default: 200]
* `--train-to INTEGER`: Diagonals to fit (default: `--width`)
* `-s, --slowdown FLOAT`: dLEM slowdown constant (per-bin)  [default: 0.025]
* `--importance-power FLOAT`: Within-diagonal emphasis on high-mass bins  [default: 0.0]
* `--diag-weight-power FLOAT`: Between-diagonal weighting by mass^power  [default: 1.0]
* `--n-opt INTEGER`: Adam optimization steps  [default: 1500]
* `--monitor TEXT`: Checkpoint to select: mse|corr|loss|symm|symm2  [default: symm2]
* `--ctcf-tsv PATH`: Comp-table TSV (`dlem.load_ctcf_tsv` format) to report CTCF alignment against
* `--plot`: Save a data-vs-prediction PNG (`prediction.png`)
* `--plot-start INTEGER`: Plot window start (bins)  [default: 0]
* `--plot-span INTEGER`: Plot window span (bins); default: `--train-to`
* `-v, --verbose`: Print optimization progress
* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

This is a single-chromosome/region tool -- it does not do genome-wide batch fitting,
bigWig track output, or predicted-`.cool` output; use the `dlem` library directly (see
the Quick start tutorial) for anything beyond one region.

**Example** (against the bundled example data, from the repo root):

```console
$ dlem docs/data/example_chr10.cool /tmp/out \
    --chrom ref_region --resolution 10000 --width 700 --train-to 200 \
    --ctcf-tsv docs/data/example_ctcf.tsv --plot --plot-span 200
```
