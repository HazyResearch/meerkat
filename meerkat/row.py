from meerkat.columns.deferred.base import DeferredCell


class Row(dict):
    def __call__(self):
        # FIXME(Sabri): Need to actually implement this based on the DAG of the deferred
        # cells for effiency.
        return Row(
            {
                name: cell() if isinstance(cell, DeferredCell) else cell
                for name, cell in self.items()
            }
        )
