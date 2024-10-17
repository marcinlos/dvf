import numpy as np


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
