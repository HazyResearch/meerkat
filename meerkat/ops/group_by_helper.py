from abc import ABC, abstractmethod

from meerkat import DataPanel


class BaseGroupBy(ABC):


    def __init__(self, indices, data, by, keys) -> None:
        self.indices = indices
        self.data = data
        self.by = by
        self.keys = keys


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
        query = self.keys
        d = {}
        d[query] = means
        if isinstance(self.by, str):
            d[self.by] = labels

            # call dp.data.reorder(list_of_string)
            return DataPanel(d)
        else:
            # self.by is a list
            if len(self.by) == 1:
                d[self.by[0]] = labels
            else:
                unzipped = list(zip(*labels))
                for i, l in enumerate(unzipped):
                    d[self.by[i]] = l
                
            return DataPanel(d)
    

