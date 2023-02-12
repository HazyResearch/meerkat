import pytest

import meerkat as mk
from meerkat.mixins.reactifiable import ReactifiableMixin


class Foo(ReactifiableMixin):
    def __init__(self, x):
        self.x = x

    @mk.gui._reactive()
    def add(self, y):
        return self.x + y

    @property
    def my_x(self):
        return self.x

    @classmethod
    @mk.gui._reactive()
    def name(cls):
        return cls.__name__

    @staticmethod
    @mk.gui._reactive()
    def static():
        return 1

    def sub(self, y):
        return self.x - y


def test_reactive_setter_inplace():
    """Setting the .reactive property should be in-place."""
    foo = Foo(1)
    foo2 = foo.react()
    foo3 = foo2.no_react()

    assert id(foo) == id(foo2) == id(foo3)


@pytest.mark.parametrize("react", [True, False])
@pytest.mark.parametrize("attr", ["x", "my_x"])
def test_attributes(react: bool, attr: str):
    foo = Foo(1)
    if react:
        foo = foo.react()

    x = getattr(foo, attr)
    assert x == 1
    assert (not react) ^ isinstance(x, mk.gui.Store)


@pytest.mark.parametrize("react", [True, False])
def test_instance_method(react: bool):
    y = mk.gui.Store(4)

    foo = Foo(1)
    if react:
        foo = foo.react()

    x = foo.add(y)
    assert x == 5
    assert isinstance(x, int)
    assert (not react) ^ isinstance(x, mk.gui.Store)

    if react:
        assert len(y.inode.trigger_children) == 1
        assert y.inode.trigger_children[0].obj.fn.__name__ == "add"
    else:
        assert y.inode is None


def test_class_method():
    """
    Class methods that are decorated with @reactive should always
    be reactive by default. This is because the class does not have
    a react flag that can be used to determine whether the method
    should be reactive or not.
    """
    name = Foo.name()
    assert isinstance(name, mk.gui.Store)
    assert name == "Foo"


def test_static_method():
    """
    Static methods that are decorated with @reactive should always
    be reactive by default. This is because the class does not have
    a react flag that can be used to determine whether the method
    should be reactive or not.
    """
    static = Foo.static()
    assert isinstance(static, mk.gui.Store)
    assert static == 1
