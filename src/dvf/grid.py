from __future__ import annotations

from collections.abc import Iterator
from enum import Flag, auto
from typing import TYPE_CHECKING, Any

import numpy as np

from dvf._util import pad_rank, pad_rank_as
from dvf.domain import Box

if TYPE_CHECKING:
    import numpy.typing as npt


class Edge(Flag):
    NONE = 0
    TOP = auto()
    BOTTOM = auto()
    LEFT = auto()
    RIGHT = auto()
    ALL = TOP | BOTTOM | LEFT | RIGHT


class Grid:
    """
    N-dimensional regular grid.

    Parameters
    ----------
    domain : Box
        Domain the grid is defined on.
    ns : int
        Number of subintervals of the grid in each dimension.

    Attributes
    ----------
    domain : Box
        Domain the grid is defined on.
    n : ndarray
        Number of subintervals of the grid in each dimension.
    h : ndarray
        Step size, i.e. the distance between any two points adjacent in
        each dimension.
    shape : tuple of int
        Number of points in the grid in each dimension.
    """

    def __init__(self, domain: Box, *ns: int) -> None:
        if domain.ndim != len(ns):
            raise ValueError(
                "Number of grid sizes does not match domain dimension "
                f"({len(ns)} given, dim = {domain.ndim})"
            )

        self.domain = domain
        self.n = np.asarray(ns)
        self.h = self.domain.edge_lengths / self.n
        self.shape = tuple(self.n + 1)

        # build an array of grid points
        coords = [
            slice(a, b, 1j * size)
            for (a, b), size in zip(self.domain.spans, self.shape, strict=True)
        ]
        self._points = np.mgrid[*coords]

    @property
    def ndim(self) -> int:
        """Number of grid dimensions."""
        return self.domain.ndim

    @property
    def size(self) -> int:
        """Number of grid points."""
        return np.prod(self.shape)

    @property
    def indices(self) -> Iterator[tuple[int, ...]]:
        """
        Iterator over grid indices.

        Each iteration produces a tuple of indices. Dimensions are iterated
        over in the order from last to first.

        Examples
        --------
        >>> grid = Grid(Box(x=(0, 1), y=(0, 1)), 2, 1)
        >>> for idx in grid.indices:
        ...     print(idx)
        (0, 0)
        (0, 1)
        (1, 0)
        (1, 1)
        (2, 0)
        (2, 1)
        """
        return np.ndindex(self.shape)

    @property
    def points(self) -> npt.NDArray:
        """
        Return an array of grid points.

        Returns
        -------
        coords : ndarray (N, k1,...,kN)
            Array of coordinates of grid points, where ``N`` is the grid
            dimension, and ``kj`` denotes the number of points in ``j``-th
            dimension. The first index selects the coordinate (``x``,
            ``y``, ...), and the others select a point of the grid.

        Examples
        --------
        >>> grid = Grid(Box(x=(0, 1), y=(0, 1)), 1, 2)
        >>> grid.points
        array([[[0. , 0. , 0. ],
                [1. , 1. , 1. ]],
        <BLANKLINE>
               [[0. , 0.5, 1. ],
                [0. , 0.5, 1. ]]])
        """
        return self._points

    def __getitem__(self, indices: Any) -> npt.NDArray:
        """
        Return coordinates of grid point at given index.

        Parameters
        ----------
        idx : valid ndarray index
            The indices of points to compute coordinates of. Any object
            that can be used to index a `numpy` array of shape
            ``self.shape`` can be used here.

        Returns
        -------
        coords : ndarray
            Coordinates of the selected points. The shape of the returned
            array is ``(N, m1, m2,...,mk)``, where ``N`` is the grid
            dimension, and ``(m1, m2,...,mk)`` is the shape of the index
            arrays.

        Examples
        --------
        >>> grid = Grid(Box(x=(0, 1), y=(0, 2)), 5, 8)
        >>> grid[1, 3]
        array([0.2 , 0.75])
        >>> grid[[1, 2, 3], [3, 4, 5]]
        array([[0.2 , 0.4 , 0.6 ],
               [0.75, 1.  , 1.25]])
        >>> large = np.sum(grid.points, axis=0) > 2.5
        >>> grid[large]
        array([[0.6 , 0.8 , 0.8 , 1.  , 1.  ],
               [2.  , 1.75, 2.  , 1.75, 2.  ]])
        >>> grid[np.ix_([0, -1], [0, -1])]
        array([[[0., 0.],
                [1., 1.]],
        <BLANKLINE>
               [[0., 2.],
                [0., 2.]]])
        """
        if not isinstance(indices, tuple):
            indices = (indices,)

        return self.points[:, *indices]

    def index_valid(self, indices: tuple[npt.ArrayLike, ...]) -> npt.NDArray[np.bool_]:
        """
        Check if indices are within bounds.

        Parameters
        ----------
        indices : tuple of array_like
            A tuple of integer arrays, one for each grid dimension.

        Returns
        -------
        ndarray of bool
            An array of boolean values indicating whether the corresponding
            index is within the grid bounds.
        """
        idx = np.asanyarray(indices)

        lower_ok = idx >= 0
        upper_ok = idx <= pad_rank_as(self.n, idx)

        return np.all(lower_ok & upper_ok, axis=0)

    @property
    def cell_volume(self) -> float:
        """Volume of a single grid cell."""
        return np.prod(self.h)

    def ravel_index(self, indices: tuple[npt.ArrayLike, ...]) -> npt.NDArray:
        """
        Linearize (flatten) grid point indices.

        The order is such that the last dimension index changes fastest.
        This is compatible with the order of indices generated by
        ``indices``. Linear indices produced by this method may be used to
        index flattened ``points`` array, in the sense that::

            flattened = grid.points.reshape(grid.ndim, -1)
            grid[*idx] == flattened[grid.ravel_index(idx)]

        for any valid index ``idx``.
        """
        return np.ravel_multi_index(indices, self.shape)  # type: ignore[arg-type,return-value]

    def boundary(self, edges: Edge = Edge.ALL) -> Iterator[tuple[int, ...]]:
        """
        Enumerate indices of boundary points.

        Each boundary edge includes the corners. Indices produced are
        unique, no point is repeated. No guarantees are given with regards
        to the order.
        """
        if Edge.TOP in edges:
            yield from ((i, 0) for i in range(self.n[0]))
            if Edge.RIGHT not in edges:
                yield (self.n[0], 0)

        if Edge.BOTTOM in edges:
            if Edge.LEFT not in edges:
                yield (0, self.n[1])
            yield from ((i, self.n[1]) for i in range(1, self.n[0] + 1))

        if Edge.LEFT in edges:
            if Edge.TOP not in edges:
                yield (0, 0)
            yield from ((0, j) for j in range(1, self.n[1] + 1))

        if Edge.RIGHT in edges:
            yield from ((self.n[0], j) for j in range(self.n[1]))
            if Edge.BOTTOM not in edges:
                yield (self.n[0], self.n[1])

    def facet_area_vector(self, indices: tuple[npt.ArrayLike, ...]) -> npt.NDArray:
        r"""
        Return area vectors for given boundary points.

        For points that do not lie at the grid boundary, zero vector is
        returned.

        Parameters
        ----------
        indices : tuple of array_like
            A tuple of integer arrays, one for each grid dimension.

        Returns
        -------
        array_vectors : ndarray
            Area vectors for the selected points. The shape of the returned
            array is ``(N, m1, m2,...,mk)``, where ``N`` is the grid
            dimension, and ``(m1, m2,...,mk)`` is the shape of the index
            arrays.

        Notes
        -----
        An area vector for a surface is a vector perpendicular to it, with
        a magnitude equal to its area. It can be understood as :math:`dS =
        \hat{n}d\sigma`, where :math:`\hat{n}` is the surface unit normal,
        and :math:`d\sigma` is the surface measure.

        In a discrete setting, area vector for a boundary point is defined
        in a way that makes the discrete version of the divergence theorem
        work:

            av = grid.facet_area_vector
            bd = grid.boundary()
            integrate(div(F, "+")) == sum(dot(F(x), av(x)) for x in bd)
        """
        idx = np.asanyarray(indices)

        at_lower_bd = idx == 0
        at_upper_bd = idx == pad_rank_as(self.n, idx)

        direction = -1 * at_lower_bd + 1 * at_upper_bd
        area = self.cell_volume / self.h
        return direction * pad_rank_as(area, idx)

    def facet_normal(self, indices: tuple[npt.ArrayLike, ...]) -> npt.NDArray:
        """
        Return unit normal vectors for given boundary points.

        For points that do not lie at the grid boundary, zero vector is
        returned.

        Parameters
        ----------
        indices : tuple of array_like
            A tuple of integer arrays, one for each grid dimension.

        Returns
        -------
        normal_vectors : ndarray
            Normal vectors for the selected points. The shape of the
            returned array is ``(N, m1, m2,...,mk)``, where ``N`` is the
            grid dimension, and ``(m1, m2,...,mk)`` is the shape of the
            index arrays.

        See Also
        --------
        facet_area_vector
        """
        v = self.facet_area_vector(indices)
        length = np.linalg.norm(v, axis=0)
        nonzero = length != 0
        return np.divide(v, length, where=nonzero)

    def facet_area(self, indices: tuple[npt.ArrayLike, ...]) -> npt.NDArray:
        """
        Return boundary area for given boundary points.

        For points that do not lie at the grid boundary, zero is returned.

        Parameters
        ----------
        indices : tuple of array_like
            A tuple of integer arrays, one for each grid dimension.

        Returns
        -------
        areas : ndarray
            Areas for the selected dpoints. The shape of the returned
            array is the same as the shaep of the index arrays.

        See Also
        --------
        facet_area_vector
        """
        v = self.facet_area_vector(indices)
        return np.linalg.norm(v, axis=0)

    def adjacent(self, index: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
        idx = np.asanyarray(index)
        nb = [-1, 0, 1]
        offsets = np.meshgrid(*[nb] * self.ndim)
        idx_ = pad_rank(idx, idx.ndim + self.ndim)
        indices = idx_ + offsets
        ok = self.index_valid(indices) & np.any(indices != idx_, axis=0)

        for i in np.moveaxis(indices[:, ok], 0, -1):
            yield tuple(i)
