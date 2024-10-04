import functools
import operator
from enum import Flag, auto

import numpy as np


class Edge(Flag):
    NONE = 0
    TOP = auto()
    BOTTOM = auto()
    LEFT = auto()
    RIGHT = auto()
    ALL = TOP | BOTTOM | LEFT | RIGHT


class Grid:
    def __init__(self, n):
        self.n = n

    @property
    def indices(self):
        for i in range(self.n + 1):
            for j in range(self.n + 1):
                yield i, j

    @property
    def points(self):
        xs = np.linspace(0.0, 1.0, self.n + 1)
        ys = np.linspace(0.0, 1.0, self.n + 1)
        return np.meshgrid(xs, ys, indexing="ij")

    def point(self, idx):
        i, j = idx
        h = self.h
        return (i * h, j * h)

    def index_valid(self, idx):
        i, j = idx
        return 0 <= i <= self.n and 0 <= j <= self.n

    @property
    def shape(self):
        return (self.n + 1, self.n + 1)

    @property
    def size(self):
        return (self.n + 1) ** 2

    @property
    def h(self):
        return 1 / self.n

    def ravel_index(self, idx):
        """
        Linearize the index.

        The order is such that the `x` index changes fastest.
        """
        return np.ravel_multi_index(idx, self.shape, order="F")

    def boundary(self, edges=Edge.ALL):
        """
        Enumerate indices of boundary points.

        Each boundary edge includes the corners. Indices produced are unique,
        no point is repeated. No guarantees are given with regards to the order.
        """
        if Edge.TOP in edges:
            yield from ((i, 0) for i in range(self.n))
            if Edge.RIGHT not in edges:
                yield (self.n, 0)

        if Edge.BOTTOM in edges:
            if Edge.LEFT not in edges:
                yield (0, self.n)
            yield from ((i, self.n) for i in range(1, self.n + 1))

        if Edge.LEFT in edges:
            if Edge.TOP not in edges:
                yield (0, 0)
            yield from ((0, j) for j in range(1, self.n + 1))

        if Edge.RIGHT in edges:
            yield from ((self.n, j) for j in range(self.n))
            if Edge.BOTTOM not in edges:
                yield (self.n, self.n)

    def boundary_normal(self, idx):
        out = np.zeros(2)
        i, j = idx

        if i == 0:
            out[0] = -1
        elif i == self.n:
            out[0] = 1

        if j == 0:
            out[1] = -1
        elif j == self.n:
            out[1] = 1

        return out

    def adjacent(self, idx):
        i, j = idx
        for a in (i - 1, i, i + 1):
            for b in (j - 1, j, j + 1):
                pair = (a, b)
                if pair != idx and self.index_valid(pair):
                    yield pair


def _lift(op, *fs):
    def composite(*args, **kwargs):
        fvals = tuple(f(*args, **kwargs) for f in fs)
        return op(*fvals)

    return composite


def _ensure_grid_equality(first, *other):
    for f in other:
        if f.grid is not first.grid:
            raise ValueError(f"Grids do not match: {f.grid} =/= {first.grid}")


def apply_to_gridfun(op, f, *fs):
    _ensure_grid_equality(f, *fs)
    fun = _lift(op, f, *fs)
    return GridFunction(fun, f.grid)


def lift_to_gridfun(op):
    return functools.partial(apply_to_gridfun, op)


class GridFunction:
    def __init__(self, fun, grid):
        self.fun = fun
        self.grid = grid

    def __call__(self, i, j):
        return self.fun(i, j)

    def tabulate(self):
        sample = self.fun(0, 0)
        out_shape = self.grid.shape + np.shape(sample)
        out = np.empty(out_shape)

        for idx in np.ndindex(*self.grid.shape):
            out[*idx, ...] = self.fun(*idx)

        return out

    @staticmethod
    def from_function(fun, grid):
        def index_fun(i, j):
            x, y = grid.point((i, j))
            return fun(x, y)

        return GridFunction(index_fun, grid)

    @staticmethod
    def from_array(data, grid):
        def index_fun(i, j):
            return data[i, j]

        return GridFunction(index_fun, grid)

    @staticmethod
    def const(val, grid):
        return GridFunction(lambda i, j: val, grid)

    @staticmethod
    def coords(grid):
        x = GridFunction.from_function(lambda x, y: x, grid)
        y = GridFunction.from_function(lambda x, y: y, grid)
        return x, y

    def _binop(self, op, other):
        if not isinstance(other, GridFunction):
            other = GridFunction.const(other, self.grid)

        return apply_to_gridfun(op, self, other)

    def _rev_binop(self, op, other):
        def reversed_op(a, b):
            return op(b, a)

        return self._binop(reversed_op, other)

    def __add__(self, other):
        return self._binop(operator.add, other)

    def __sub__(self, other):
        return self._binop(operator.sub, other)

    def __mul__(self, other):
        return self._binop(operator.mul, other)

    def __truediv__(self, other):
        return self._binop(operator.truediv, other)

    def __pow__(self, other):
        return self._binop(operator.pow, other)

    def __radd__(self, other):
        return self._rev_binop(operator.add, other)

    def __rsub__(self, other):
        return self._rev_binop(operator.sub, other)

    def __rmul__(self, other):
        return self._rev_binop(operator.mul, other)

    def __rtruediv__(self, other):
        return self._rev_binop(operator.truediv, other)

    def __rpow__(self, other):
        return self._rev_binop(operator.pow, other)

    def __neg__(self):
        return apply_to_gridfun(operator.neg, self)

    def __abs__(self):
        return apply_to_gridfun(operator.abs, self)


sin = lift_to_gridfun(np.sin)
cos = lift_to_gridfun(np.cos)
tan = lift_to_gridfun(np.tan)
asin = lift_to_gridfun(np.asin)
acos = lift_to_gridfun(np.acos)
atan = lift_to_gridfun(np.atan)
sinh = lift_to_gridfun(np.sinh)
cosh = lift_to_gridfun(np.cosh)
tanh = lift_to_gridfun(np.tanh)
asinh = lift_to_gridfun(np.asinh)
acosh = lift_to_gridfun(np.acosh)
atanh = lift_to_gridfun(np.atanh)
exp = lift_to_gridfun(np.exp)
log = lift_to_gridfun(np.log)
sqrt = lift_to_gridfun(np.sqrt)


def integrate(f):
    grid = f.grid
    h = grid.h
    return h**2 * sum(f(*idx) for idx in grid.indices)


def integrate_bd(f):
    grid = f.grid
    h = grid.h
    return h * sum(f(*idx) for idx in grid.boundary())


def delta(i, j, grid):
    def fun(p, q):
        return 1.0 if (p, q) == (i, j) else 0.0

    return GridFunction(fun, grid)


def _axis_index(axis):
    match axis:
        case "x":
            return 0
        case "y":
            return 1
        case _:
            raise ValueError(f"Invalid axis: {axis}")


def shift(f, axis, offset=1):
    axis_idx = _axis_index(axis)

    def fun(ix, iy):
        idx = np.array([ix, iy])
        idx[axis_idx] += offset

        if not f.grid.index_valid(idx):
            return np.zeros_like(f(ix, iy))

        return f(*idx)

    return GridFunction(fun, f.grid)


def diff(f, axis, mode):
    axis_idx = _axis_index(axis)

    match mode:
        case "+":
            sign = -1
            offset = +1
        case "-":
            sign = +1
            offset = -1
        case _:
            raise ValueError(f"Invalid differentiation mode: {mode}")

    def fun(ix, iy):
        idx = np.array([ix, iy])
        idx[axis_idx] += offset

        if not f.grid.index_valid(idx):
            return np.zeros_like(f(ix, iy))

        current = f(ix, iy)
        other = f(*idx)

        return sign * (current - other) / f.grid.h

    return GridFunction(fun, f.grid)


def dx(f, mode):
    return diff(f, "x", mode)


def dy(f, mode):
    return diff(f, "y", mode)


def nabla(f, mode):
    combine = lift_to_gridfun(lambda *xs: np.stack(xs, axis=-1))
    return combine(dx(f, mode), dy(f, mode))


def div(f, mode):
    # np.linalg.trace sums over the last two indices, unlike np.trace
    return apply_to_gridfun(np.linalg.trace, nabla(f, mode))


def norm(f, kind):
    return np.sqrt(product(f, f, kind))


def product(f, g, kind):
    match kind:
        case "h":
            fun = f * g
        case "grad_h":
            dot = lift_to_gridfun(np.dot)
            fun = dot(nabla(f, "+"), nabla(g, "+"))
        case _:
            raise ValueError(f"Invalid norm: {kind}")

    return integrate(fun)
