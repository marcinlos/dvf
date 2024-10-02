import operator
from enum import Flag, auto

import numpy as np


class Edge(Flag):
    NONE = 0
    TOP = auto()
    BOTTOM = auto()
    LEFT = auto()
    RIGHT = auto()


ALL_EDGES = Edge.TOP | Edge.BOTTOM | Edge.LEFT | Edge.RIGHT


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

    def boundary(self, edges=ALL_EDGES):
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

    def adjacent(self, idx):
        i, j = idx
        for a in (i - 1, i, i + 1):
            for b in (j - 1, j, j + 1):
                pair = (a, b)
                if pair != idx and self.index_valid(pair):
                    yield pair


def lift(op, *fs):
    def composite(*args, **kwargs):
        fvals = tuple(f(*args, **kwargs) for f in fs)
        return op(*fvals)

    return composite


class GridFunction:
    def __init__(self, fun, grid):
        self.fun = fun
        self.grid = grid

    def __call__(self, i, j):
        return self.fun(i, j)

    def tabulate(self):
        return np.fromfunction(self.fun, self.grid.shape, dtype=np.intp)

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

    def _ensure_same_grid(self, other):
        if self.grid is not other.grid:
            raise ValueError(f"Grids do not match: {self.grid} =/= {other.grid}")

    def _binop(self, op, other):
        if not isinstance(other, GridFunction):
            other = GridFunction.const(other, self.grid)

        self._ensure_same_grid(other)
        fun = lift(op, self.fun, other.fun)
        return GridFunction(fun, self.grid)

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
        return GridFunction(lift(operator.neg, self.fun), self.grid)

    def __abs__(self):
        return GridFunction(lift(operator.abs, self.fun), self.grid)
