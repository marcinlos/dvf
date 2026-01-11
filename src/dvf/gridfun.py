import functools
import operator

import numpy as np

from dvf._util import get_unique_grid


def _lift(op, *fs):
    def composite(*args, **kwargs):
        fvals = tuple(f(*args, **kwargs) for f in fs)
        return op(*fvals)

    return composite


def apply_to_gridfun(op, *fs):
    grid = get_unique_grid(f.grid for f in fs)
    fun = _lift(op, *fs)
    return GridFunction(fun, grid)


def lift_to_gridfun(op):
    return functools.partial(apply_to_gridfun, op)


class GridFunction:
    def __init__(self, fun, grid):
        self.fun = fun
        self.grid = grid

    def __call__(self, *idx):
        return self.fun(*idx)

    def tabulate(self):
        zero = (0,) * self.grid.ndim
        sample = self.fun(*zero)
        out_shape = np.shape(sample) + self.grid.shape
        out = np.empty(out_shape)

        for idx in np.ndindex(*self.grid.shape):
            out[..., *idx] = self.fun(*idx)

        return out

    @staticmethod
    def from_function(fun, grid):
        def index_fun(*idx):
            xs = grid[idx]
            return fun(*xs)

        return GridFunction(index_fun, grid)

    @staticmethod
    def from_array(data, grid):
        def index_fun(*idx):
            return data[..., *idx]

        return GridFunction(index_fun, grid)

    @staticmethod
    def const(val, grid):
        return GridFunction(lambda *idx: val, grid)

    @staticmethod
    def coords(grid):
        def coord(i):
            return GridFunction.from_function(lambda *xs: xs[i], grid)

        return tuple(coord(i) for i in range(grid.ndim))

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


def as_tensor(a):
    def grids(a):
        if isinstance(a, GridFunction):
            yield a.grid
        try:
            for x in a:
                yield from grids(x)
        except TypeError:
            pass

    grid_list = list(grids(a))

    def fun(*idx):
        def evaluate(a):
            if isinstance(a, GridFunction):
                return a(*idx)

            try:
                return tuple(evaluate(x) for x in a)
            except TypeError:
                # assume it is a constant
                return a

        return np.array(evaluate(a))

    grid = grid_list[0]
    return GridFunction(fun, grid)


def random_function(grid, shape=(), bc=None, dist=np.random.rand):
    data = dist(*shape, *grid.shape)

    if bc is not None:
        data.flat[bc] = 0

    return GridFunction.from_array(data, grid)


def delta(idx, /, equal=1.0, not_equal=0.0):
    def fun(*p):
        return equal if p == idx else not_equal

    return fun


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

    def fun(*idx):
        new_idx = np.copy(idx)
        new_idx[axis_idx] += offset

        if f.grid.index_valid(new_idx):
            return f(*new_idx)
        else:
            return np.zeros_like(f(*idx))

    return GridFunction(fun, f.grid)


def diff(f, axis, mode):
    axis_idx = _axis_index(axis)
    h = f.grid.h[axis_idx]

    match mode:
        case "+":
            offset = +1
        case "-":
            offset = -1
        case _:
            raise ValueError(f"Invalid differentiation mode: {mode}")

    def fun(*idx):
        other_idx = np.copy(idx)
        other_idx[axis_idx] += offset

        current = f(*idx)

        if f.grid.index_valid(other_idx):
            other = f(*other_idx)
            return (other - current) / (offset * h)
        else:
            return np.zeros_like(current)

    return GridFunction(fun, f.grid)


def Dx(f, mode):
    return diff(f, "x", mode)


def Dy(f, mode):
    return diff(f, "y", mode)


def grad(f, mode):
    combine = lift_to_gridfun(lambda *xs: np.stack(xs, axis=-1))
    return combine(Dx(f, mode), Dy(f, mode))


def div(f, mode):
    # np.linalg.trace sums over the last two indices, unlike np.trace
    return apply_to_gridfun(np.linalg.trace, grad(f, mode))
