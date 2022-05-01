from abc import ABC, abstractmethod
from meerkat.columns.numpy_column import NumpyArrayColumn
from meerkat import DataPanel


class AbstractColumnGroupBy(ABC):

    def __init__(self, indices, data, by, keys) -> None:
        self.indices = indices
        self.data = data
        self.by = by
        self.keys = keys

    @abstractmethod
    def mean(self):
        raise NotImplementedError()

class NumPyArrayGroupBy(AbstractColumnGroupBy):

    def mean(self):

        assert(isinstance(self.keys, str) )
        means = []
        s_indices = list(self.indices.keys())
        s_indices.sort()
        labels = s_indices
        for key in s_indices:
            indices_l = self.indices[key]
            appropriate_slice = self.data[indices_l]
            mean_slice = appropriate_slice.mean()
            means.append(mean_slice)
        query = self.keys
        return DataPanel({"by_column": labels, query : means})

class TensorGroupBy(AbstractColumnGroupBy):
    pass


class SeriesGroupBy(AbstractColumnGroupBy):
    pass


class ArrowGroupBy(AbstractColumnGroupBy):
    pass

class NumpyArrayColumn(NumpyArrayColumn):
    def to_group_by(self, indices, by, keys) -> NumPyArrayGroupBy:
        return NumPyArrayGroupBy(indices, self.data, by, keys)