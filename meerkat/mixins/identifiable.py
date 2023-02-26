from uuid import uuid4

from meerkat.tools.utils import classproperty

_MK_ID_PREFIX = "__mkid__"


class IdentifiableMixin:
    """Mixin for classes, to give objects an id.

    This class must use _self_{attribute} for all attributes since it
    will be mixed into the wrapt.ObjectProxy class, which requires this
    naming convention for it to work.
    """

    _self_identifiable_group: str

    def __init__(self, id: str = None, *args, **kwargs):
        super(IdentifiableMixin, self).__init__(*args, **kwargs)
        self._set_id(id=id)

    @property
    def id(self):
        return self._self_id

    # Note: this is set to be a classproperty, so that we can access either
    # cls.identifiable_group or self.identifiable_group.

    # If this is changed to a property, then we can only access
    # self.identifiable_group, and cls._self_identifiable_group but not
    # cls.identifiable_group. This is fine if something breaks, but
    # be careful to change cls.identifiable_group to
    # cls._self_identifiable_group everywhere.
    @classproperty
    def identifiable_group(self):
        return self._self_identifiable_group

    def _set_id(self, id: str = None):
        # get uuid as str
        if id is None:
            self._self_id = _MK_ID_PREFIX + uuid4().hex
        else:
            # TODO: Have objects that need special treatment for being
            # detected in endpoint (e.g. Store, dataframes) implement their
            # own set id method to assert that id is a meerkat id.
            # assert is_meerkat_id(id)
            self._self_id = id

        from meerkat.state import state

        state.identifiables.add(self)

    @classmethod
    def from_id(cls, id: str):
        # TODO(karan): make sure we're using this everywhere and it's not
        # being redefined in subclasses
        from meerkat.state import state

        return state.identifiables.get(id=id, group=cls.identifiable_group)

    @staticmethod
    def prepend_meerkat_id_prefix(id: str) -> str:
        return _MK_ID_PREFIX + id


def is_meerkat_id(id: str) -> bool:
    return id.startswith(_MK_ID_PREFIX)
