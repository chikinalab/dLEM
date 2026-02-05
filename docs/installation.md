# Installation
## Using a managed environment
- Requirements: Python 3.11–3.13, recent pip/setuptools/wheel.
- Create and activate an env (pick one):
  - Conda: `conda create -n dlem-env python=3.12 -y && conda activate dlem-env`
  - Mamba/Micromamba: `mamba create -n dlem-env python=3.12 -y && mamba activate dlem-env`
  - python -m venv: `python3 -m venv .venv && source .venv/bin/activate`
  - virtualenv: `python3 -m pip install --upgrade virtualenv && python3 -m virtualenv .venv && source .venv/bin/activate`
- Upgrade tooling: `python -m pip install --upgrade pip setuptools wheel`
- Option A: local checkout
  - Clone and enter the repo: `git clone https://github.com/chikinalab/dLEM.git && cd dLEM`
  - Install: `python -m pip install .`
- Option B: direct from GitHub (no local clone)
  - `python -m pip install "git+https://github.com/chikinalab/dLEM.git"`

## From Pypi
`pip install dlem`

## From source
`pip install git+https://github.com/chikinalab/dLEM.git`

## Container
`docker run ghcr.io/dborgesr/dlem:latest`