


from abc import ABC, abstractmethod

class BaseGroupBy(ABC):
    def __init__(self, indices, data, by, keys) -> None:
        self.indices = indices
        self.data = data
        self.by = by
        self.keys = keys

    def mean(self):
        # inputs: self.indices are a dictionary of {
        #   labels : [indices]
        # }
        labels = list(self.indices.keys())

        # sorting them so that they appear in a nice order.
        labels.sort()


        print("Mean", labels)

        # Means will be a list of dictionaries where each element in the dict

        means = []
        for label in labels:
            indices_l = self.indices[label]
            relevant_rows_where_by_is_label = self.data[indices_l]
            m = relevant_rows_where_by_is_label.mean()
            means.append(m)

        from meerkat.datapanel import DataPanel

        # Create data panel as a list of rows.
        out = DataPanel(means)


        assert isinstance(self.by, list)

        # Add the by columns.
        if len(labels) > 0:
            if isinstance(labels[0], tuple):
                columns = list(zip(*labels))

                for i, col in enumerate(self.by):
                    out[col] = columns[i]
            else:
                # This is the only way that this can occur.
                assert(len(self.by) == 1)
                col = self.by[0]
                out[col] = labels
        return out





