# Discrete Variational Formulation

## Dependencies

- Python version >= 3.12
- [uv](https://docs.astral.sh/uv/)

## Installation

Clone the git repository:
```bash
git clone https://github.com/marcinlos/dvf.git
```
Download dependencies:
```bash
uv sync # no pytorch
```
```bash
uv sync --extra gpu # pytorch on GPU
```
```bash
uv sync --extra cpu # pytorch on CPU
```
Recreate notebook files using Jupytext:
```bash
uv run jupytext --sync scripts/*.py
```

## Running the examples

Start the Jupyter server:
```
uv run jupyter lab
```
and run the code in the notebooks.
