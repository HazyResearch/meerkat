from meerkat.tools.identifier import Identifier


class IdentifierMixin:
    def __init__(self, identifier: Identifier, *args, **kwargs):
        super(IdentifierMixin, self).__init__(*args, **kwargs)

        # Identifier for the object
        self._identifier = (
            Identifier(self.__class__.__name__) if not identifier else identifier
        )

    @property
    def identifier(self):
        return self._identifier

    @property
    def id(self):
        return self.identifier
