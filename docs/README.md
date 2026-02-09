# Differentiable Loop Extrusion Model
This package provides functions to train, predict, and evaluate Differentiable Loop Extrusion Model (DLEM) on HiC/Micro-C experiments formated as (m)cool files.

## Installation:
Requires Python 3.11-3.13
`pip install dlem-jax`

## Tutorials:
* [CLI Usage](cli_usage.ipynb)
* [Synthetic Data](synthetic_data.ipynb)
* [Train and use model w/ Python API](train_analyze_example.ipynb)

## References:
* [CLI Reference](cli_reference.md)
* [API Reference](api_reference.md)
* [Core Reference](core_reference.md)

## Contributing
Install dev environment w/ Pixi:

`pixi install --environment dev` 

Contributing using Docker:

`docker-compose build`

`docker-compose up dlem-dev --watch`