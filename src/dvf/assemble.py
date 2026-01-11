import numpy as np


def _relevant_points(idx, grid):
    # assuming first-order differential operators, it is enough to consider
    # the cross-shaped stenicil
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


def _build_dof_map(space, idx, offsets):
    idx = np.asanyarray(idx)

    def basis(off):
        i = tuple(idx + off)
        return tuple(space.basis_at(i)) if space.grid.index_valid(i) else ()

    return {off: basis(off) for off in offsets}


def assemble_optimized(form, matrix, trial_fun, test_fun, links):
    U = trial_fun.space
    V = test_fun.space

    grid = U.grid
    offsets_trial = {off for ((off, _), _) in links}
    offsets_test = {off for (_, (off, _)) in links}
    trial_funs = {info for (info, _) in links}

    from collections import defaultdict

    test_for_trial = defaultdict(list)
    for trial_data, test_data in links:
        test_for_trial[trial_data].append(test_data)

    for p in grid.indices:
        trial_basis = _build_dof_map(U, p, offsets_trial)
        test_basis = _build_dof_map(V, p, offsets_test)

        for u_off, i in trial_funs:
            if not trial_basis[u_off]:
                continue
            u = trial_basis[u_off][i]

            for v_off, j in test_for_trial[u_off, i]:
                if not test_basis[v_off]:
                    continue
                v = test_basis[v_off][j]

                trial_fun.assign(u)
                test_fun.assign(v)
                matrix[v.index, u.index] += grid.cell_volume * form(*p)
