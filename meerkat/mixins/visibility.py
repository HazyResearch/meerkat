from typing import Optional, Sequence

import numpy as np


class VisibilityMixin:
    """This mixin assumes the existence of a `self.data` property."""

    _visible_rows: Optional[np.ndarray] = None

    def __init__(self, *args, **kwargs):
        super(VisibilityMixin, self).__init__(*args, **kwargs)

    @property
    def visible_rows(self):
        return self._visible_rows

    @visible_rows.setter
    def visible_rows(self, indices: Optional[Sequence]):
        """Set the visible rows of the object."""
        if indices is not None and len(indices):
            assert min(indices) >= 0 and max(indices) < len(self), (
                f"Ensure min index {min(indices)} >= 0 and"
                f" max index {max(indices)} < {len(self)}."
            )

        if self._visible_rows is None:
            if indices is None:
                self._visible_rows = None
            else:
                self._visible_rows = np.array(indices, dtype=int)
        else:
            if indices is None:
                # do nothing – keep old visible_roows
                pass
            else:
                self._visible_rows = self._visible_rows[np.array(indices, dtype=int)]

        # TODO (sabri): look at this for virtual column
        # # Identify that `self` corresponds to a DataPanel
        # if hasattr(self, "_data") and isinstance(self._data, Mapping):
        #     # Need to set visible_rows for all columns, not just visible ones
        #     for column in self._data.values():
        #         column.visible_rows = indices

    def _remap_index(self, index):
        # TODO: lazy import needed to avoid circular dependency, this is should be
        # avoided by defining a "real" abstract class above `AbstractColumn`
        from meerkat.columns.abstract import AbstractColumn

        if isinstance(index, int):
            return self.visible_rows[index].item()
        elif isinstance(index, slice):
            return self.visible_rows[index]
        elif isinstance(index, str):
            return index
        elif (isinstance(index, tuple) or isinstance(index, list)) and len(index):
            return self.visible_rows[index]
        elif isinstance(index, np.ndarray) and len(index.shape) == 1:
            return self.visible_rows[index]
        elif isinstance(index, AbstractColumn):
            return self.visible_rows[index]
        else:
            raise TypeError(
                "Object of type {} is not a valid index".format(type(index))
            )
