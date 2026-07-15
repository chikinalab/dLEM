

# Documentation
[Documentation](https://chikinalab.github.io/dLEM)


<!--
Copyright (C) 2026 Diego Borges-Rivera <dborgesrivera@clarku.edu>

SPDX-License-Identifier: MIT
-->

# dLEM

---

<!-- markdownlint-disable MD033 -->

<table>
    <tr>
      <td>Downloads</td>
      <td>
        <a href="https://github.com/users/dborgesr/packages/container/package/dlem">
          <img src="https://img.shields.io/badge/github-container-blue" alt="GithubRegistry">
        </a>
        &nbsp
      </td>
    </tr>
    <tr>
      <td>Documentation</td>
      <td>
        <a href="https://chikinalab.github.io/dLEM">
          <img src="https://readthedocs.org/projects/dLEM/badge/?version=latest" alt="Documentation">
        </a>
      </td>
    </tr>
    <tr>
      <td>License</td>
      <td>
        <a href="https://github.com/chikinalab/dLEM/blob/main/LICENSE">
          <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
        </a>
      </td>
    </tr>
    <tr>
      <td>CI</td>
      <td>
        &nbsp
        <a href="https://github.com/chikinalab/dLEM/blob/main/.github/workflows/publish.yml">
          <img src="https://github.com/chikinalab/dLEM/actions/workflows/publish.yml/badge.svg" alt="Build Dockerfile Status">
        </a>
      </td>
    </tr>
</table>

<!-- markdownlint-enable MD033 -->

---

# Differentiable Loop Extrusion Model
This package provides functions to train, predict, and evaluate a Differentiable Loop Extrusion Model (DLEM) on HiC/Micro-C experiments. Take chromatin conformation data in (m)cool format and calculate L and R cohesin rate parameters.

## Installation
Requires Python 3.11–3.13
`pip install dlem-jax`

### From source
`pip install git+https://github.com/chikinalab/dLEM.git`

Refer to the [Installation](https://chikinalab.github.io/dLEM/installation) section in the documentation for more information and how to contribute.

## Quickstart

**`dLEM`**

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
* `--help`: Show this message and exit

This is a single-chromosome/region tool -- it does not do genome-wide batch fitting,
bigWig track output, or predicted-`.cool` output; use the `dlem` library directly (see
`docs/quick_start.ipynb`) for anything beyond one region.

**Example** (against the bundled example data):

```console
$ dlem docs/data/example_chr10.cool /tmp/out \
    --chrom ref_region --resolution 10000 --width 700 --train-to 200 \
    --ctcf-tsv docs/data/example_ctcf.tsv --plot --plot-span 200
```

### Docker usage
`docker run ghcr.io/dborgesr/dlem:latest`

## Contributing
`git clone https://github.com/chikinalab/dLEM.git`

### Using local pixi
`pixi install -e`

### Using Docker
`docker-compose build`

`docker-compose up --watch`

## Citing

If you use dLEM or any of its language bindings in your research, please cite the following publication:

Tina Subic, Tŭgrul Balcı, Kristina Perevoshchikova, Geoffrey Fudenberg, Maria Chikina, Mechanistic Genome Folding at Scale through the Differentiable Loop Extrusion Model
_Biorxiv_, [https://www.biorxiv.org/content/10.1101/2025.10.17.682904v1](https://www.biorxiv.org/content/10.1101/2025.10.17.682904v1)

<details>
<summary>BibTex</summary>

```bibtex
@article{dlem,
    author = {Tina Subic, Tŭgrul Balcı, Kristina Perevoshchikova, Geoffrey Fudenberg, Maria Chikina},
    title = "{dlem: diffrentiable loop extrusion model for chromatin looping data}",
    journal = {Biorxiv},
    volume = {40},
    number = {7},
    pages = {btae408},
    year = {2024},
    month = {06},
    issn = {1367-4811},
    doi = {10.1101/2025.10.17.682904},
    url = {https://doi.org/10.1101/2025.10.17.682904},
    eprint = {https://www.biorxiv.org/content/10.1101/2025.10.17.682904v1.full.pdf},
}
```

</details>
