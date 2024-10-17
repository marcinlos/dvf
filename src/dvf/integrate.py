import numpy as np

from dvf.gridfun import grad, lift_to_gridfun


def integrate(f):
    grid = f.grid
    return grid.cell_volume * sum(f(*idx) for idx in grid.indices)


def integrate_bd(f):
    grid = f.grid
    return sum(f(*idx) * grid.facet_area(idx) for idx in grid.boundary())


def norm(f, kind):
    return np.sqrt(product(f, f, kind))


def product(f, g, kind):
    match kind:
        case "h":
            fun = f * g
        case "grad_h":
            dot = lift_to_gridfun(np.dot)
            fun = dot(grad(f, "+"), grad(g, "+"))
        case _:
            raise ValueError(f"Invalid norm: {kind}")

    return integrate(fun)
