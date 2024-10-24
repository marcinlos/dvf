import typing
from typing import NamedTuple


class Interval(NamedTuple):
    """
    Closed interval with real endpoints.

    Parameters
    ----------
    start : float
        The left endpoint.
    end : float
        The right endpoint.

    Examples
    --------
    >>> s = Interval(1, 2.5)
    >>> s.length
    1.5
    >>> 1.8 in s
    True
    """

    start: float
    end: float

    @property
    def length(self) -> float:
        """Length of the interval."""
        return self.end - self.start

    def __contains__(self, x: object) -> bool:
        """
        Check if a number is inside the interval.

        Parameters
        ----------
        x : object
            Number to check.

        Returns
        -------
        bool
            `True` if `x` is inside the interval, `False` otherwise.
        """
        x = typing.cast(float, x)
        return self.start <= x <= self.end


class Box:
    """
    Hyperrectangle in arbitrary dimensional Euclidean space.

    The hyperrectangle is represented as a cartesian product of intervals.
    Each interval is associated with a named dimension (axis).

    Parameters
    ----------
    dims : tuple of two floats or Interval

    Examples
    --------
    >>> cube = Box(x=(0, 1), y=(-1, 2), z=(1, 3))
    >>> cube["y"]
    Interval(start=-1, end=2)
    >>> cube[2]
    Interval(start=1, end=3)
    >>> cube.ndim
    3
    >>> cube.axes
    ('x', 'y', 'z')
    """

    def __init__(self, **dims: tuple[float, float]) -> None:
        self._dims = {name: Interval(*span) for name, span in dims.items()}
        self._by_index = tuple(self._dims.values())

    def __getitem__(self, idx: int | str) -> Interval:
        """
        Retrieve the span along axis given by position or name.

        Parameters
        ----------
        idx : int or str
            Axis specified either by its position in order during
            construction, or by its name.

        Returns
        -------
        Interval
            Span of the hyperrectangle along given axis.
        """
        match idx:
            case str():
                return self._dims[idx]
            case _:
                return self._by_index[idx]

    @property
    def axes(self) -> tuple[str, ...]:
        """Names of the dimensions (axes)."""
        return tuple(self._dims)

    @property
    def ndim(self) -> int:
        """Dimension of the hyperrectangle."""
        return len(self._dims)
