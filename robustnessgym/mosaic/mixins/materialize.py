class MaterializationMixin:
    def __init__(self, materialize: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Flag that determines whether the object will be materialized or not
        self._materialize = materialize

    @property
    def materialize(self):
        return self._materialize
