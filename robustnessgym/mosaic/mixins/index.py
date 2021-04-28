class IndexableMixin:
    def __init__(self, n: int, *args, **kwargs):
        super(IndexableMixin, self).__init__(*args, **kwargs)

        # Index associated with each element
        self._index = [str(i) for i in range(n)]

    @property
    def index(self):
        return self._index
