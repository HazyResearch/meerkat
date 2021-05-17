class MaterializationMixin:
    def __init__(self, materialize: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Flag that determines whether the object will be materialized or not
        self._materialize = materialize

    @property
    def lz(self):
        return _LazyIndexer(self)


class _LazyIndexer:
    def __init__(self, obj: object):
        self.obj = obj

    def __getitem__(self, index):
        return self.obj._get(index, materialize=False)

    def __len__(self):
        return len(self.obj)
