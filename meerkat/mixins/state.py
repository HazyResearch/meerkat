from dataclasses import dataclass
from typing import Collection, Dict, Union

from meerkat.tools.utils import nested_map


@dataclass
class StateClass:
    """An internal class to store the state of an object alongside its
    associated class."""

    klass: type
    state: object


class StateDictMixin:
    def __init__(self, *args, **kwargs):
        super(StateDictMixin, self).__init__(*args, **kwargs)

    @classmethod
    def _state_keys(cls) -> Collection:
        return NotImplemented

    @classmethod
    def _assert_state_keys(cls, state: Dict) -> None:
        """Assert that a state contains all required keys."""
        if cls._state_keys() is NotImplemented:
            return

        assert (
            set(state.keys()) == cls._state_keys()
        ), f"State must contain all state keys: {cls._state_keys()}."

    def get_state(self) -> Dict:
        """Get the internal state of the object.

        For complex objects (e.g. Spacy Doc), this should return a
        compressed representation of the object.
        """

        def _apply_get_state(obj):
            if hasattr(obj, "get_state"):
                return StateClass(**{"klass": type(obj), "state": obj.get_state()})
            else:
                return obj

        if self._state_keys() is NotImplemented:
            state = nested_map(
                _apply_get_state,
                {key: getattr(self, key) for key in self.__dict__.keys()},
            )
        else:
            state = nested_map(
                _apply_get_state,
                {key: getattr(self, key) for key in self._state_keys()},
            )
            self._assert_state_keys(state)

        return state

    @classmethod
    def from_state(cls, state: Union[Dict, StateClass], *args, **kwargs) -> object:
        """Load the object from state."""

        def _apply_from_state(obj_: Union[Dict, StateClass]):
            if isinstance(obj_, StateClass):
                return obj_.klass.from_state(obj_.state, *args, **kwargs)
            else:
                return obj_

        if isinstance(state, StateClass):
            assert (
                state.klass == cls
            ), f"`state` has klass={state.klass} but `from_state` was called by {cls}."

            # Extract the state dict
            state = state.state

        # Apply from state recursively
        state = nested_map(_apply_from_state, state)

        # Check that all keys are present
        cls._assert_state_keys(state)

        # Create a new object and update its state
        try:
            obj = cls(**state)
        except TypeError:
            obj = cls()
        obj.__dict__.update(state)

        return obj
