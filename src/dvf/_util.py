import numpy as np


def pad_rank(a, rank):
    a = np.asanyarray(a)
    missing = rank - a.ndim
    new_shape = a.shape + (1,) * missing
    return a.reshape(new_shape)


def pad_rank_as(a, other):
    rank = np.ndim(other)
    return pad_rank(a, rank)


def get_unique_grid(grids):
    grid_list = list(grids)

    if not grid_list:
        return ValueError("Empty list")

    first, *other = grid_list

    for g in other:
        if g is not first:
            raise ValueError(f"Grids do not match: {g} =/= {first}")

    return first
