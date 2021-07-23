from typing import Any, Dict


class CloneableMixin:
    def copy(self, **kwargs) -> object:
        new_data = self._copy_data()
        return self._clone(data=new_data, **kwargs)

    def view(self, **kwargs) -> object:
        new_data = self._view_data()
        return self._clone(data=new_data, **kwargs)

    def _clone(self, data: object = None, **kwargs):
        if data is None:
            data = self._data

        state = self._get_state()

        if kwargs:
            state.update(kwargs)

        obj = self.__class__.__new__(self.__class__)
        obj._set_data(data)
        obj._set_state(state)
        return obj

    def _copy_data(self) -> object:
        raise NotImplementedError

    def _view_data(self) -> object:
        raise NotImplementedError

    def _get_state(self) -> dict:
        return {key: getattr(self, key) for key in self._state_keys()}

    def _set_state(self, state: dict):
        self.__dict__.update(state)

    def _set_data(self, data):
        self._data = data
