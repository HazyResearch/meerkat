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
        d = {}
        d[query] = means
        if isinstance(self.by, str):
            d[self.by] = labels
            return DataPanel(d)
        else:

            if len(self.by) == 1:
                d[self.by[0]] = labels
            else:
                unzipped = list(zip(*labels))
                for i, l in enumerate(unzipped):
                    d[self.by[i]] = l
                
            return DataPanel(d)


class TensorGroupBy(AbstractColumnGroupBy):
    pass


class SeriesGroupBy(AbstractColumnGroupBy):
    pass


class ArrowGroupBy(AbstractColumnGroupBy):
    pass

class NumpyArrayColumn(NumpyArrayColumn):
    def to_group_by(self, indices, by, keys) -> NumPyArrayGroupBy:
        return NumPyArrayGroupBy(indices, self.data, by, keys)