

# Documentation
[Documentation](https://DreschLab.github.io/dlem)


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
        <a href="https://anaconda.org/bioconda/">
          <img src="https://img.shields.io/conda/vn/bioconda/hictk?label=bioconda&logo=Anaconda" alt="Bioconda">
        </a>
        &nbsp
        <a href="https://hub.docker.com/r/paulsengroup/hictk">
          <img src="https://img.shields.io/docker/pulls/paulsengroup/hictk" alt="DockerHub">
        </a>
        &nbsp
      </td>
    </tr>
    <tr>
      <td>Documentation</td>
      <td>
        <a href="https://dlem.readthedocs.io/">
          <img src="https://readthedocs.org/projects/dlem/badge/?version=latest" alt="Documentation">
        </a>
      </td>
    </tr>
    <tr>
      <td>License</td>
      <td>
        <a href="https://github.com/paulsengroup/hictk/blob/main/LICENSE">
          <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
        </a>
      </td>
    </tr>
    <tr>
      <td>CI</td>
      <td>
        <a href="https://github.com/paulsengroup/hictk/actions/workflows/ubuntu-ci.yml">
          <img src="https://github.com/paulsengroup/hictk/actions/workflows/ubuntu-ci.yml/badge.svg" alt="Ubuntu CI Status">
        </a>
        &nbsp
        <a href="https://github.com/paulsengroup/hictk/actions/workflows/macos-ci.yml">
          <img src="https://github.com/paulsengroup/hictk/actions/workflows/macos-ci.yml/badge.svg" alt="macOS CI Status">
        </a>
        &nbsp
        <a href="https://github.com/paulsengroup/hictk/actions/workflows/windows-ci.yml">
          <img src="https://github.com/paulsengroup/hictk/actions/workflows/windows-ci.yml/badge.svg" alt="Windows CI Status">
        </a>
        &nbsp
        <a href="https://github.com/paulsengroup/hictk/actions/workflows/build-dockerfile.yml">
          <img src="https://github.com/paulsengroup/hictk/actions/workflows/build-dockerfile.yml/badge.svg" alt="Build Dockerfile Status">
        </a>
      </td>
    </tr>
</table>

<!-- markdownlint-enable MD033 -->

---

# Differentiable Loop Extrusion Model
This package provides functions to train, predict, and evaluate a Differentiable Loop Extrusion Model (DLEM) on HiC/Micro-C experiments.
## Features

### Supported formats

The CLI application and python library is capable of reading and writing files in the following formats:

<!-- markdownlint-disable MD033 -->

| Format | Revision   | Read | Write           |
| ------ | ---------- | ---- | --------------- |
| .cool  | v1-3 (all) | ✅   | ✅ <sup>1</sup> |
| .mcool | v1-2 (all) | ✅   | ✅ <sup>2</sup> |

<small><small>

<sup>1</sup> v3 only\
<sup>2</sup> v2 only\

</small></small>

<!-- markdownlint-enable MD033 -->

### Supported operations

- Take chromatin looping data and define L and R cohesin parameters

All the above operations can be performed on both Cooler and .hic files and yield identical results.

## Installation

dLEM can be installed from pip using \
Refer to the [Installation](https://dlem.readthedocs.io/en/stable/installation.html) section in the documentation for more information.

`uv pip install dlem`

## Quickstart

# `dLEM`

dLEM: differentiable Loop Extrusion Model

**Usage**:

```console
$ dlem [OPTIONS] DNA_INTS OUTPUT_LOCATION
```

**Arguments**:

* `DNA_INTS`: Chromatin input dataset in the form of a .cool file  [required]
* `OUTPUT_LOCATION`: Location of output directory, will be created if not existing  [required]

**Options**:

* `--device [gpu|cpu|tpu]`: Select computation device: gpu, cpu, or tpu  [default: cpu]
* `-r, --region TEXT`: Train on a specific chromosomal region (ex: chr16:51906425-55560999)
* `-b, --bed TEXT`: Plot and output tsv for all regions in BED file,
3 columns , ex:
* `--chromosomes COMMA_LIST_PARSER`: Comma-separated list of chromosomes to train on (e.g., &#x27;chr1,chr2,chr3&#x27;)
* `--all`: Run dLEM across all chromosomes in input cool file
* `-l, --resolution INTEGER`: Resolution to pull .cool file from .mcool file, must exist
* `--start-diag INTEGER`: Diagonal offset where updates start during training  [default: 5]
* `--train-rows INTEGER`: Number of diagonals to use during training dLEM  [default: 170]
* `--steps INTEGER`: Number of steps for training  [default: 10]
* `-w, --window-size INTEGER`: Column window size for slowdown fitting  [default: 3000]
* `-s, --window-step INTEGER`: Step between slowdown-fitting windows  [default: 200]
* `-d, --decay-extent INTEGER`: Number of band rows to use when fitting slowdown  [default: 500]
* `-i, --iterations INTEGER`: Number of iterations to fit the model over  [default: 300]
* `-lr, --lr FLOAT`: Change speed of parameter tuning, useful if loss is too fast/slow or model not converging  [default: 0.01]
* `-pm, --prediction-mode [mse|corr|final]`: Which parameter set to use for predictions  [default: mse]
* `-esm, --early-stop-metric [mse|corr|none]`: Metric for early stopping  [default: mse]
* `--debug`: Enable debug mode
* `-pr, --prediction-rows INTEGER`: Number of diagonals to predict with using dLEM model  [default: 700]
* `-c, --output-cool`: Output the prediction as a .cool file
* `-t, --output-tracks`: Output the parameters as genomic tracks in BED and BigWig format
* `--norm`: Normalize the predicted band. When set geometric mean is used, otherwise no normalization
* `-p, --plot`: Plot fits to compare to input
* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.


## Development

`docker-compose build`
`docker-compose up --watch`

Refer to the
[Quickstart (CLI)](cli_usage.md) and
[CLI Reference](cli_reference.md)
sections in the documentation for more details.

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