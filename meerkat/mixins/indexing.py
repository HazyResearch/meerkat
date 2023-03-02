from pyparsing import Mapping


class MaterializationMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def mz(self):
        return _MaterializeIndexer(self)


class _MaterializeIndexer:
    def __init__(self, obj: object):
        self.obj = obj

    def __getitem__(self, index):
        return self.obj._get(index, materialize=True)

    def __len__(self):
        return len(self.obj)

    @property
    def loc(self):
        return _LocIndexer(self.obj, materialize=True)


class _LocIndexer:
    def __init__(self, obj: object, materialize: bool = False):
        self.obj = obj
        self.materialize = materialize

    def __getitem__(self, keyidx):
        return self.obj._get_loc(keyidx, materialize=self.materialize)

    def __setitem__(
        self,
        index,
        value,
    ):
        if isinstance(index, tuple):
            keyidx, column = index
            self.obj._set_loc(keyidx=keyidx, column=column, value=value)
        else:
            if isinstance(value, Mapping):
                raise ValueError(
                    "Must pass mapping if column is not specified on __setitem__."
                )
            for column in self.obj.columns:
                self.obj._set_loc(
                    keyidx=index,
                    column=column,
                    value=value[column],
                )

    def __len__(self):
        return len(self.obj)

    @property
    def mz(self):
        return _LocIndexer(self.obj, materialize=True)


class IndexerMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def loc(self):
        return _LocIndexer(self)
