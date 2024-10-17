from enum import Flag, auto

import numpy as np

from dvf._util import pad_rank, pad_rank_as


class Edge(Flag):
    NONE = 0
    TOP = auto()
    BOTTOM = auto()
    LEFT = auto()
    RIGHT = auto()
    ALL = TOP | BOTTOM | LEFT | RIGHT


class Grid:
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny
        self.n = np.array([nx, ny])
        self.h = 1 / self.n
        self.shape = tuple(self.n + 1)
        self.ndim = 2

    @property
    def indices(self):
        return np.ndindex(self.shape)

    @property
    def points(self):
        coords = [np.s_[0.0 : 1.0 : 1j * size] for size in self.shape]
        return np.mgrid[*coords]

    def point(self, idx):
        return idx * pad_rank_as(self.h, idx)

    def index_valid(self, idx):
        idx = np.asanyarray(idx)

        lower_ok = idx >= 0
        upper_ok = idx <= pad_rank_as(self.n, idx)

        out = np.all(lower_ok & upper_ok, axis=0)

        if out.size == 1:
            return out.item()
        else:
            return out

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def cell_volume(self):
        return np.prod(self.h)

    def ravel_index(self, idx):
        """
        Linearize the index.

        The order is such that the `y` index changes fastest.
        """
        return np.ravel_multi_index(idx, self.shape)

    def boundary(self, edges=Edge.ALL):
        """
        Enumerate indices of boundary points.

        Each boundary edge includes the corners. Indices produced are unique,
        no point is repeated. No guarantees are given with regards to the order.
        """
        if Edge.TOP in edges:
            yield from ((i, 0) for i in range(self.nx))
            if Edge.RIGHT not in edges:
                yield (self.nx, 0)

        if Edge.BOTTOM in edges:
            if Edge.LEFT not in edges:
                yield (0, self.ny)
            yield from ((i, self.ny) for i in range(1, self.nx + 1))

        if Edge.LEFT in edges:
            if Edge.TOP not in edges:
                yield (0, 0)
            yield from ((0, j) for j in range(1, self.ny + 1))

        if Edge.RIGHT in edges:
            yield from ((self.nx, j) for j in range(self.ny))
            if Edge.BOTTOM not in edges:
                yield (self.nx, self.ny)

    def _facet_vector(self, idx):
        idx = np.asanyarray(idx)

        at_lower_bd = idx == 0
        at_upper_bd = idx == pad_rank_as(self.n, idx)

        direction = -1 * at_lower_bd + 1 * at_upper_bd
        area = self.cell_volume / self.h
        return direction * area

    def facet_normal(self, idx):
        v = self._facet_vector(idx)
        return v / np.linalg.norm(v)

    def facet_area(self, idx):
        v = self._facet_vector(idx)
        return np.linalg.norm(v)

    def adjacent(self, idx):
        idx = np.asanyarray(idx)
        nb = [-1, 0, 1]
        offsets = np.meshgrid(*([nb] * self.ndim))
        idx_ = pad_rank(idx, idx.ndim + self.ndim)
        indices = idx_ + offsets
        ok = self.index_valid(indices) & np.any(indices != idx_, axis=0)

        for i in np.moveaxis(indices[:, ok], 0, -1):
            yield tuple(i)
