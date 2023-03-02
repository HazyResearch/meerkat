from dataclasses import dataclass

from meerkat.interactive.node import NodeMixin
from meerkat.mixins.blockable import BlockableMixin
from meerkat.mixins.identifiable import IdentifiableMixin
from meerkat.provenance import ProvenanceMixin


@dataclass
class StateClass:
    """An internal class to store the state of an object alongside its
    associated class."""

    klass: type
    state: object


class CloneableMixin:
    def __init__(self, *args, **kwargs):
        super(CloneableMixin, self).__init__(*args, **kwargs)

    @classmethod
    def _state_keys(cls) -> set:
        """"""
        raise NotImplementedError()

    @classmethod
    def _clone_keys(cls) -> set:
        return set()

    def copy(self, **kwargs) -> object:
        new_data = self._copy_data()
        return self._clone(data=new_data)

    def view(self) -> object:
        return self._clone()

    def _clone(self, data: object = None, **kwargs):
        if data is None:
            if isinstance(self, BlockableMixin) and self.is_blockable():
                data = self._pack_block_view()
            else:
                data = self._view_data()

        state = self._get_state(clone=True)
        state.update(kwargs)

        obj = self.__class__.__new__(self.__class__)
        obj._set_state(state)
        obj._set_data(data)

        if isinstance(self, ProvenanceMixin):
            # need to create a node for the object
            obj._init_node()

        if isinstance(self, IdentifiableMixin):
            obj._set_id()

        if isinstance(self, NodeMixin):
            obj._set_inode()
            # obj._set_children()

        from meerkat.dataframe import DataFrame

        if isinstance(obj, DataFrame):
            if obj.primary_key_name not in obj:
                # need to reset the primary key if we remove the column
                obj.set_primary_key(None, inplace=True)

        return obj

    def _copy_data(self) -> object:
        raise NotImplementedError

    def _view_data(self) -> object:
        return self.data

    def _get_state(self, clone: bool = False) -> dict:
        state = {key: getattr(self, key) for key in self._state_keys()}
        if clone:
            state.update({key: getattr(self, key) for key in self._clone_keys()})
        return state

    def _set_state(self, state: dict):
        self.__dict__.update(state)
