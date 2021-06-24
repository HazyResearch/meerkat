from __future__ import annotations

import abc
import copy as pycopy
from typing import TYPE_CHECKING

from meerkat.provenance import ProvenanceObjNode

if TYPE_CHECKING:
    from meerkat import AbstractColumn, DataPanel


class AbstractCopyMixin(abc.ABC):
    def __init__(self, *args, **kwargs):
        super(AbstractCopyMixin, self).__init__(*args, **kwargs)

    def _copy(self, view: bool = False):
        return NotImplemented

    def view(self) -> object:
        """When called on a column, returns a new column with the same
        underlying data, but distinct views into that data (e.g.
        `AbstractColumn.visible_rows`, `AbstractColumn._collate_fn`, and
        `DataPanel.visible_columns`). Consider the following example: ```

            >>> column = ListColumn([0,0,0])
            >>> view = column.view()
            >>> view[0] = 10
            >>> print(column[0] == view[0])
            True
            >>> view.visible_rows = [1,2]
            >>> print(column[0] == view[0])
            False
        ```
        When called on a DataPanel, returns a new DataPanel containing views of all the
        columns in the original. Note: the underlying dictionary holding the columns
        differs between the original and view DataPanels.
        Consider the following example:
        ```
            >>> dp = DataPanel.from_csv(...)
            >>> view = dp.view()
            >>> view.add_column("new_col", ListColumn([0] * len(dp)))
            >>> print("new_col" in dp)
            False
        ```
        """
        return self._copy(view=True)

    def copy(self) -> object:
        """Returns columns with a shallow copy of the underlying data."""
        return self._copy(view=False)

    def deepcopy(self) -> object:
        """Return a deepcopy of the column or DataPanel."""
        return pycopy.deepcopy(self)


class ColumnCopyMixin(AbstractCopyMixin):
    def _copy(self, view: bool = False) -> AbstractColumn:
        """Return a copy of the object."""
        state = {}
        for k, v in self.__dict__.items():
            if k == "_data" and view:
                # don't copy the underlying data of the column if creating view
                state[k] = v
            elif k != "_node":
                state[k] = pycopy.copy(v)

        try:
            obj = self.__class__(**state)
        except (TypeError, ValueError):
            # use `__new__` to instantiate a bare class, in case __init__ does work
            # we do not want to repeat on copy
            obj = self.__class__.__new__(self.__class__)

        obj.__dict__ = state
        obj._node = ProvenanceObjNode(obj)

        return obj


class DataPanelCopyMixin(AbstractCopyMixin):
    def _copy(self, view: bool = False) -> DataPanel:
        state = {}
        for k, v in self.__dict__.items():

            # TODO: make this a nested map to cover sequences
            # TODO (sabri): fix __new__ missing data for numpy array
            if k == "_data":
                state[k] = {
                    kp: pycopy.copy(vp)
                    if not hasattr(vp, "copy")
                    else vp._copy(view=view)
                    for kp, vp in v.items()
                }
            elif k != "_node":
                state[k] = pycopy.copy(v)

        try:
            obj = self.__class__(**state)
        except TypeError:
            # use `__new__` to instantiate a bare class, in case __init__ does work
            # we do not want to repeat on copy
            obj = self.__class__.__new__(self.__class__)
        obj.__dict__ = state
        obj._node = ProvenanceObjNode(obj)
        return obj
