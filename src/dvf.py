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


def _pad_rank(a, rank):
    a = np.asanyarray(a)
    missing = rank - a.ndim
    new_shape = a.shape + (1,) * missing
    return a.reshape(new_shape)


def _pad_rank_as(a, other):
    rank = np.ndim(other)
    return _pad_rank(a, rank)


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
        return idx * _pad_rank_as(self.h, idx)

    def index_valid(self, idx):
        idx = np.asanyarray(idx)

        lower_ok = idx >= 0
        upper_ok = idx <= _pad_rank_as(self.n, idx)

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
        at_upper_bd = idx == _pad_rank_as(self.n, idx)

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
        idx_ = _pad_rank(idx, idx.ndim + self.ndim)
        indices = idx_ + offsets
        ok = self.index_valid(indices) & np.any(indices != idx_, axis=0)

        for i in np.moveaxis(indices[:, ok], 0, -1):
            yield tuple(i)


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
            xs = grid.point(idx)
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


def integrate(f):
    grid = f.grid
    return grid.cell_volume * sum(f(*idx) for idx in grid.indices)


def integrate_bd(f):
    grid = f.grid
    return sum(f(*idx) * grid.facet_area(idx) for idx in grid.boundary())


def delta(idx, /, equal=1.0, not_equal=0.0):
    def fun(p, q):
        return equal if (p, q) == idx else not_equal

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


class FunctionVariable(GridFunction):
    def __init__(self, space):
        super().__init__(self._eval, space.grid)
        self.space = space
        self._source_fun = None

    def _eval(self, *idx):
        if self._source_fun is None:
            return self.space.zero
        else:
            return self._source_fun(*idx)

    def assign(self, fun):
        self._source_fun = fun

    def clear(self):
        self.assign(None)


class FunctionSpace:
    def __init__(self, grid):
        self.grid = grid
        self.zero = 0
        self.zero_fun = GridFunction.const(self.zero, grid)

    @property
    def dim(self):
        return self.grid.size

    @property
    def shape(self):
        return ()

    def basis_at(self, point_idx):
        f = delta(point_idx)
        fun = GridFunction(f, self.grid)
        idx = self.grid.ravel_index(point_idx)
        fun.index = idx
        yield fun

    def trial_function(self):
        return FunctionVariable(self)

    def test_function(self):
        return self.trial_function()


class VectorFunctionSpace:
    def __init__(self, grid, components):
        self.grid = grid
        self.components = components

        self.zero = np.zeros(components)
        self.zero_fun = GridFunction.const(self.zero, grid)
        self.local_basis = np.identity(components)

    @property
    def dim(self):
        return self.grid.size * self.components

    @property
    def shape(self):
        return (self.components,)

    def basis_at(self, point_idx):
        offset = self.grid.ravel_index(point_idx)

        for i in range(self.components):
            one = self.local_basis[i, :]
            f = delta(point_idx, one, self.zero)
            fun = GridFunction(f, self.grid)

            base = i * self.grid.size
            fun.index = base + offset
            fun.component = (i,)
            yield fun

    def trial_function(self):
        return FunctionVariable(self)

    def test_function(self):
        return self.trial_function()


class TensorFunctionSpace:
    def __init__(self, grid, shape):
        self.grid = grid
        self.shape = shape
        self.zero = np.zeros(self.shape)
        self.zero_fun = GridFunction.const(self.zero, grid)

        self.local_basis = np.zeros(shape + shape)
        for idx in np.ndindex(*shape):
            self.local_basis[*idx, *idx] = 1

    @property
    def dim(self):
        return self.grid.size * np.prod(self.shape)

    def basis_at(self, point_idx):
        offset = self.grid.ravel_index(point_idx)

        for comp_idx in np.ndindex(*self.shape):
            one = self.local_basis[*comp_idx, ...]
            f = delta(point_idx, one, self.zero)
            fun = GridFunction(f, self.grid)

            base = np.ravel_multi_index(comp_idx, self.shape) * self.grid.size
            fun.index = base + offset
            fun.component = comp_idx
            yield fun

    def trial_function(self):
        return FunctionVariable(self)

    def test_function(self):
        return self.trial_function()


class CompositeBasisFunction:
    def __init__(self, components, index):
        self.components = components
        self.index = index


class CompositeFunctionVariable:
    def __init__(self, space, components):
        self.space = space
        self.components = components

    def assign(self, fun):
        for c, f in zip(self.components, fun.components, strict=True):
            c.assign(f)

    def clear(self):
        for c in self.components():
            c.clear()


class CompositeFunctionSpace:
    def __init__(self, *spaces):
        _ensure_grid_equality(*spaces)
        self.spaces = spaces
        self.grid = spaces[0].grid
        self.zero_funs = tuple(space.zero_fun for space in spaces)

    @property
    def dim(self):
        return sum(U.dim for U in self.spaces)

    def basis_at(self, point_idx):
        offset = 0

        for i, space in enumerate(self.spaces):
            for f in space.basis_at(point_idx):
                components = list(self.zero_funs)
                components[i] = f
                index = offset + f.index
                yield CompositeBasisFunction(components, index)

            offset += space.dim

    def trial_function(self):
        trial_funs = tuple(space.trial_function() for space in self.spaces)
        return CompositeFunctionVariable(self, trial_funs)

    def test_function(self):
        test_funs = tuple(space.test_function() for space in self.spaces)
        return CompositeFunctionVariable(self, test_funs)

    def combine_dofs(self, *dof_sets):
        if len(dof_sets) != len(self.spaces):
            raise ValueError(
                f"Invalid number of DoF sets ({len(dof_sets)} =/= {len(self.spaces)})"
            )

        offset = 0
        combined_dofs = []

        for dofs, space in zip(dof_sets, self.spaces, strict=True):
            shifted = np.array(dofs) + offset
            combined_dofs.extend(shifted)
            offset += space.dim

        return combined_dofs


def select_dofs(space, condition, invert=False):
    def gen():
        for idx in space.grid.indices:
            for u in space.basis_at(idx):
                result = u(*idx) * condition(*idx)
                matches = np.any(result)

                if matches != invert:
                    yield u.index

    return np.array(list(gen()), dtype=np.intp)


def remove_dofs(a, dofs=None, *, trial_dofs=None, test_dofs=None):
    match a.ndim:
        case 1:
            if trial_dofs is not None:
                raise ValueError("Cannot specify trial DoFs for vector")

            if test_dofs is not None:
                raise ValueError("Cannot specify test DoFs for vector")

            if dofs is None:
                raise ValueError("DoFs to remove not specified")

            return np.delete(a, dofs)

        case 2:
            if dofs is not None:
                if trial_dofs is not None:
                    raise ValueError("Cannot mix `dofs` and `trial_dofs`")

                if test_dofs is not None:
                    raise ValueError("Cannot mix `dofs` and `test_dofs`")

                trial_dofs_ = dofs
                test_dofs_ = dofs
            else:
                if trial_dofs is None and test_dofs is None:
                    raise ValueError("DoFs to remove not specified")

                trial_dofs_ = trial_dofs or ()
                test_dofs_ = test_dofs or ()

            b = np.delete(a, test_dofs_, axis=0)
            return np.delete(b, trial_dofs_, axis=1)

        case _:
            raise ValueError(f"Tensors of rank {a.ndim} are not supported")


def reinsert_dofs(a, dofs, value=0):
    # kind of a hack - `np.insert` inserts items at the indices
    # of the vector as it is, not as it was before removing them,
    # so we need to modify them accordingly
    dof_indices_in_a = np.sort(dofs) - np.arange(len(dofs))
    return np.insert(a, dof_indices_in_a, value)


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
