from typing import Mapping, Optional, Sequence

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
        if indices is None:
            self._visible_rows = None
        else:
            if len(indices):
                assert min(indices) >= 0 and max(indices) < len(self), (
                    f"Ensure min index {min(indices)} >= 0 and "
                    f"max index {max(indices)} < {len(self)}."
                )
            if self._visible_rows is not None:
                self._visible_rows = self._visible_rows[np.array(indices, dtype=int)]
            else:
                self._visible_rows = np.array(indices, dtype=int)

        # Identify that `self` corresponds to a DataPane
        if hasattr(self, "_data") and isinstance(self._data, Mapping):
            for column in self.values():
                column.visible_rows = self._visible_rows

    def _remap_index(self, index):
        if isinstance(index, int):
            return self.visible_rows[index].item()
        elif isinstance(index, slice):
            return self.visible_rows[index].tolist()
        elif isinstance(index, str):
            return index
        elif (isinstance(index, tuple) or isinstance(index, list)) and len(index):
            return self.visible_rows[index].tolist()
        elif isinstance(index, np.ndarray) and len(index.shape) == 1:
            return self.visible_rows[index].tolist()
        else:
            raise TypeError("Invalid argument type: {}".format(type(index)))
