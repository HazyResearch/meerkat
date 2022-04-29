from abc import ABC, abstractmethod
from meerkat import DataPanel

class AbstractColumnGroupBy(ABC):

    def __init__(self, indices, data) -> None:
        self.indices = indices
        self.data = data

    @abstractmethod
    def mean(self):
        raise NotImplementedError()

class NumPyArrayGroupBy(AbstractColumnGroupBy):
    def mean(self):
        means = []
        s_indices = list(self.indices.keys())
        s_indices.sort()
        labels = s_indices
        for key in s_indices:
            indices_l = self.indices[key]
            appropriate_slice = self.data[indices_l]
            mean_slice = appropriate_slice.mean()
            means.append(mean_slice)
        return DataPanel({"by_column": labels, "data": means})

class TensorGroupBy(AbstractColumnGroupBy):
    pass


class SeriesGroupBy(AbstractColumnGroupBy):
    pass


class ArrowGroupBy(AbstractColumnGroupBy):
    pass