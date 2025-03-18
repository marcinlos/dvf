# Discrete Variational Formulation

## Dependencies

- Python version >= 3.11
- [poetry](https://python-poetry.org/) (install e.g.
  [using `pipx`](https://python-poetry.org/docs/#installing-with-pipx))

## Installation

Clone the git repository:
```
git clone https://github.com/marcinlos/dvf.git
```
Resolve and download dependencies:
```
poetry install
```
Activate the virtual environment:
```
poetry shell
```
Recreate notebook files using Jupytext:
```
jupytext --sync scripts/*.py
```

## Running the examples

Start the Jupyter server:
```
jupyter lab
```
and run the code in the notebooks.
