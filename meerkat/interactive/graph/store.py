import warnings
from typing import Any, Generic, Union

from pydantic import BaseModel, Field, ValidationError
from pydantic.fields import ModelField
from wrapt import ObjectProxy

from meerkat.interactive.graph.reactivity import reactive
from meerkat.interactive.modification import StoreModification
from meerkat.interactive.node import NodeMixin
from meerkat.interactive.types import Storeable, T
from meerkat.mixins.identifiable import IdentifiableMixin

__all__ = ["Store", "StoreFrontend", "make_store", "store_field"]


class StoreFrontend(BaseModel):
    store_id: str
    value: Any
    has_children: bool
    is_store: bool = True


# ObjectProxy must be the last base class
class Store(IdentifiableMixin, NodeMixin, Generic[T], ObjectProxy):

    _self_identifiable_group: str = "stores"

    def __init__(self, wrapped: T, backend_only: bool = False):
        super().__init__(wrapped=wrapped)
        # Set up these attributes so we can create the
        # schema and detail properties.
        self._self_schema = None
        self._self_detail = None
        self._self_value = None
        self._self_backend_only = backend_only

    @property
    def value(self):
        return self.__wrapped__

    def to_json(self):
        return self.__wrapped__

    @property
    def frontend(self):
        return StoreFrontend(
            store_id=self.id,
            value=self.__wrapped__,
            has_children=self.inode.has_children() if self.inode else False,
        )

    @property
    def detail(self):
        return f"Store({self.__wrapped__}) has id {self.id} and node {self.inode}"

    def set(self, new_value: T):
        """Set the value of the store."""
        if isinstance(new_value, Store):
            # if the value is a store, then we need to unpack so it can be sent to the
            # frontend
            new_value = new_value.__wrapped__

        mod = StoreModification(id=self.id, value=new_value)
        self.__wrapped__ = new_value
        mod.add_to_queue()

    def __repr__(self) -> str:
        return f"Store({repr(self.__wrapped__)})"

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self.__wrapped__, name)

        if callable(attr):
            # Instance method
            return reactive(attr)
        else:
            # Attribute
            return reactive(lambda store: getattr(store.__wrapped__, name))(self)

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, field: ModelField):
        if not isinstance(v, cls):
            if not field.sub_fields:
                # Generic parameters were not provided so we don't try to validate
                # them and just return the value as is
                return cls(v)
            else:
                # Generic parameters were provided so we try to validate them
                # and return a Store object
                v, error = field.sub_fields[0].validate(v, {}, loc="value")
                if error:
                    raise ValidationError(error)
                return cls(v)
        return v

    @reactive
    def __lt__(self, other):
        return super().__lt__(other)

    @reactive
    def __le__(self, other):
        return super().__le__(other)

    @reactive
    def __eq__(self, other):
        return super().__eq__(other)

    @reactive
    def __ne__(self, other):
        return super().__ne__(other)

    @reactive
    def __gt__(self, other):
        return super().__gt__(other)

    @reactive
    def __ge__(self, other):
        return super().__ge__(other)

    def __hash__(self):
        return hash(self.__wrapped__)

    @reactive
    def __nonzero__(self):
        return super().__nonzero__()

    def __bool__(self):
        # __bool__ cannot be reactive because Python expects
        # __bool__ to return a bool and not a Store.
        # This means stores cannot be used in logical statements.
        return super().__bool__()

    @reactive
    def __add__(self, other):
        return super().__add__(other)

    @reactive
    def __sub__(self, other):
        return super().__sub__(other)

    @reactive
    def __mul__(self, other):
        return super().__mul__(other)

    @reactive
    def __div__(self, other):
        return super().__div__(other)

    @reactive
    def __truediv__(self, other):
        return super().__truediv__(other)

    @reactive
    def __floordiv__(self, other):
        return super().__floordiv__(other)

    @reactive
    def __mod__(self, other):
        return super().__mod__(other)

    @reactive(nested_return=False)
    def __divmod__(self, other):
        return super().__divmod__(other)

    @reactive
    def __pow__(self, other, *args):
        return super().__pow__(other, *args)

    @reactive
    def __lshift__(self, other):
        return super().__lshift__(other)

    @reactive
    def __rshift__(self, other):
        return super().__rshift__(other)

    @reactive
    def __and__(self, other):
        return super().__and__(other)

    @reactive
    def __xor__(self, other):
        return super().__xor__(other)

    @reactive
    def __or__(self, other):
        return super().__or__(other)

    @reactive
    def __radd__(self, other):
        return super().__radd__(other)

    @reactive
    def __rsub__(self, other):
        return super().__rsub__(other)

    @reactive
    def __rmul__(self, other):
        return super().__rmul__(other)

    @reactive
    def __rdiv__(self, other):
        return super().__rdiv__(other)

    @reactive
    def __rtruediv__(self, other):
        return super().__rtruediv__(other)

    @reactive
    def __rfloordiv__(self, other):
        return super().__rfloordiv__(other)

    @reactive
    def __rmod__(self, other):
        return super().__rmod__(other)

    @reactive
    def __rdivmod__(self, other):
        return super().__rdivmod__(other)

    @reactive
    def __rpow__(self, other, *args):
        return super().__rpow__(other, *args)

    @reactive
    def __rlshift__(self, other):
        return super().__rlshift__(other)

    @reactive
    def __rrshift__(self, other):
        return super().__rrshift__(other)

    @reactive
    def __rand__(self, other):
        return super().__rand__(other)

    @reactive
    def __rxor__(self, other):
        return super().__rxor__(other)

    @reactive
    def __ror__(self, other):
        return super().__ror__(other)

    # We do not need to decorate i-methods because they call their
    # out-of-place counterparts, which are reactive
    def __iadd__(self, other):
        warnings.warn(
            f"{type(self).__name__}.__iadd__ is out-of-place. Use __add__ instead."
        )
        return self.__add__(other)

    def __isub__(self, other):
        warnings.warn(
            f"{type(self).__name__}.__isub__ is out-of-place. Use __sub__ instead."
        )
        return self.__sub__(other)

    def __imul__(self, other):
        warnings.warn(
            f"{type(self).__name__}.__imul__ is out-of-place. Use __mul__ instead."
        )
        return self.__mul__(other)

    def __idiv__(self, other):
        warnings.warn(
            f"{type(self).__name__}.__idiv__ is out-of-place. Use __div__ instead."
        )
        return self.__div__(other)

    def __itruediv__(self, other):
        warnings.warn(
            f"{type(self).__name__}.__itruediv__ is out-of-place. "
            "Use __truediv__ instead."
        )
        return self.__truediv__(other)

    def __ifloordiv__(self, other):
        warnings.warn(
            f"{type(self).__name__}.__ifloordiv__ is out-of-place. "
            "Use __floordiv__ instead."
        )
        return self.__floordiv__(other)

    def __imod__(self, other):
        warnings.warn(
            f"{type(self).__name__}.__imod__ is out-of-place. Use __mod__ instead."
        )
        return self.__mod__(other)

    def __ipow__(self, other):
        warnings.warn(
            f"{type(self).__name__}.__ipow__ is out-of-place. Use __pow__ instead."
        )
        return self.__pow__(other)

    def __ilshift__(self, other):
        warnings.warn(
            f"{type(self).__name__}.__ilshift__ is out-of-place. "
            "Use __lshift__ instead."
        )
        return self.__lshift__(other)

    def __irshift__(self, other):
        warnings.warn(
            f"{type(self).__name__}.__irshift__ is out-of-place. "
            "Use __rshift__ instead."
        )
        return self.__rshift__(other)

    def __iand__(self, other):
        warnings.warn(
            f"{type(self).__name__}.__iand__ is out-of-place. Use __and__ instead."
        )
        return self.__and__(other)

    def __ixor__(self, other):
        warnings.warn(
            f"{type(self).__name__}.__ixor__ is out-of-place. Use __xor__ instead."
        )
        return self.__xor__(other)

    def __ior__(self, other):
        warnings.warn(
            f"{type(self).__name__}.__ior__ is out-of-place. Use __or__ instead."
        )
        return self.__or__(other)

    @reactive
    def __neg__(self):
        return super().__neg__()

    @reactive
    def __pos__(self):
        return super().__pos__()

    @reactive
    def __abs__(self):
        return super().__abs__()

    @reactive
    def __invert__(self):
        return super().__invert__()

    @reactive
    def __int__(self):
        return super().__int__()

    @reactive
    def __long__(self):
        return super().__long__()

    @reactive
    def __float__(self):
        return super().__float__()

    @reactive
    def __complex__(self):
        return super().__complex__()

    @reactive
    def __oct__(self):
        return super().__oct__()

    @reactive
    def __hex__(self):
        return super().__hex__()

    @reactive
    def __index__(self):
        return super().__index__()

    @reactive
    def __len__(self):
        return super().__len__()

    @reactive
    def __contains__(self, value):
        return super().__contains__(value)

    @reactive
    def __getitem__(self, key):
        return super().__getitem__(key)

    @reactive
    def __setitem__(self, key, value):
        # Make a shallow copy of the value because this operation is not in-place.
        obj = self.__wrapped__.copy()
        obj[key] = value
        warnings.warn(f"{type(self).__name__}.__setitem__ is out-of-place.")
        return type(self)(obj, backend_only=self._self_backend_only)

    @reactive
    def __delitem__(self, key):
        obj = self.__wrapped__.copy()
        del obj[key]
        warnings.warn(f"{type(self).__name__}.__delitem__ is out-of-place.")
        return type(self)(obj, backend_only=self._self_backend_only)

    @reactive
    def __getslice__(self, i, j):
        return super().__getslice__(i, j)

    @reactive
    def __setslice__(self, i, j, value):
        obj = self.__wrapped__.copy()
        obj[i:j] = value
        warnings.warn(f"{type(self).__name__}.__setslice__ is out-of-place.")
        return type(self)(obj, backend_only=self._self_backend_only)

    @reactive
    def __delslice__(self, i, j):
        obj = self.__wrapped__.copy()
        del obj[i:j]
        warnings.warn(f"{type(self).__name__}.__delslice__ is out-of-place.")
        return type(self)(obj, backend_only=self._self_backend_only)

    # def __enter__(self):
    #     return self.__wrapped__.__enter__()

    # def __exit__(self, *args, **kwargs):
    #     return self.__wrapped__.__exit__(*args, **kwargs)

    @reactive
    def __next__(self):
        return next(self.__wrapped__)

    @reactive
    def __iter__(self):
        return iter(self.__wrapped__)


def store_field(value: str) -> Field:
    """Utility for creating a pydantic field with a default factory that
    creates a Store object wrapping the given value.

    TODO (karan): Take a look at this again. I think we might be able to
    get rid of this in favor of just passing value.
    """
    return Field(default_factory=lambda: Store(value))


def make_store(value: Union[str, Storeable]) -> Store:
    """Make a Store.

    If value is a Store, return it. Otherwise, return a
    new Store that wraps value.

    Args:
        value (Union[str, Storeable]): The value to wrap.

    Returns:
        Store: The Store wrapping value.
    """
    return value if isinstance(value, Store) else Store(value)
