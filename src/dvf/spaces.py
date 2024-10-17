import numpy as np

from dvf._util import get_unique_grid
from dvf.gridfun import GridFunction, delta


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
        self.spaces = spaces
        self.grid = get_unique_grid(s.grid for s in spaces)
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
