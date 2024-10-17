import numpy as np


def _relevant_points(idx, grid):
    # assuming first-order differential operators, it is enough to consider the
    # cross-shaped stenicil
    idx = np.asanyarray(idx)
    units = np.identity(grid.ndim, dtype=int)
    zero = np.zeros((1, grid.ndim), dtype=int)
    offsets = np.concat([units, -units, zero])
    points = idx + offsets
    ok = grid.index_valid(points.T)
    return [tuple(idx) for idx in points[ok]]


def _relevant_basis_funs(space, points):
    def gen():
        for p in points:
            yield from space.basis_at(p)

    return list(gen())


def assemble(form, matrix, trial_fun, test_fun):
    U = trial_fun.space
    V = test_fun.space

    grid = U.grid

    for p in grid.indices:
        pts = _relevant_points(p, grid)
        trial_basis = _relevant_basis_funs(U, pts)
        test_basis = _relevant_basis_funs(V, pts)

        for u in trial_basis:
            for v in test_basis:
                trial_fun.assign(u)
                test_fun.assign(v)
                matrix[v.index, u.index] += grid.cell_volume * form(*p)
