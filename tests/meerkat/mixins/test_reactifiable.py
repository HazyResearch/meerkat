import pytest

import meerkat as mk
from meerkat.mixins.reactifiable import ReactifiableMixin


class Foo(ReactifiableMixin):
    def __init__(self, x):
        self.x = x

    @mk.react()
    def add(self, y):
        return self.x + y

    # Test that the decorator can be used with or without parentheses.
    @mk.react
    def add_dec_no_parenthesis(self, y):
        return self.x + y

    # Test that react works for magic methods, when accessing them with
    # their shortcuts (e.g. foo[0]).
    @mk.react()
    def __getitem__(self, idx):
        return self.x

    # Test that properties behave like normal attribute accessing.
    # When the instance is reactive, accessing the property should be reactive.
    @property
    def my_x(self):
        return self.x

    # Python requires __len__ to return an int.
    # Because this method is not wrapped with @no_react, it will
    # return self.x, which can be a Store when `is_reactive() and self._reactive`.
    # This will raise an error.
    def __len__(self):
        return self.x

    @mk.no_react()
    def add_no_react(self, y):
        return self.x + y

    # Test if a method is not decorated with @react* decorators, then
    # it should automatically be wrapped
    def add_auto_react(self, y):
        return self.x + y

    @classmethod
    @mk.react()
    def name(cls):
        return cls.__name__

    @staticmethod
    @mk.react()
    def static():
        return 1


# TODO: Add tests for nested react/noreact funcs.


def test_reactive_setter_inplace():
    """Setting the .reactive property should be in-place."""
    foo = Foo(1)
    foo2 = foo.react()
    foo3 = foo2.react()
    foo4 = foo2.no_react()

    assert foo is foo2
    assert foo is foo3
    assert foo is foo4


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
@pytest.mark.parametrize("method", ["add", "add_dec_no_parenthesis"])
def test_instance_method_decorated(react: bool, method: str):
    y = mk.gui.Store(4)

    foo = Foo(1)
    if react:
        foo = foo.react()

    x = getattr(foo, method)(y)
    assert x == 5
    assert isinstance(x, int)
    assert (not react) ^ isinstance(x, mk.gui.Store)

    if react:
        assert len(y.inode.trigger_children) == 1
        assert y.inode.trigger_children[0].obj.fn.__name__ == method
    else:
        assert y.inode is None


@pytest.mark.parametrize("react", [True, False])
@pytest.mark.parametrize("y_as_store", [False, True])
def test_magic_method_react_shortcut_accessor(react: bool, y_as_store: bool):
    y = 1
    if y_as_store:
        y = mk.gui.Store(y)

    foo = Foo(1)
    if react:
        foo = foo.react()

    x = foo[y]
    assert x == 1
    assert isinstance(x, int)
    assert (not react) ^ isinstance(x, mk.gui.Store)
    if react and isinstance(y, mk.gui.Store):
        assert len(y.inode.trigger_children) == 1
        assert y.inode.trigger_children[0].obj.fn.__name__ == "__getitem__"
    elif isinstance(y, mk.gui.Store):
        assert y.inode is None


@pytest.mark.parametrize("react", [True, False])
def test_magic_method_not_decorated(react: bool):
    """Magic methods that are not decorated should never be reactive."""
    foo = Foo(1)
    if react:
        foo = foo.react()
        # We should see an error because we do not explicitly set no_react()
        # on __len__. This means __len__ can return a store.
        with pytest.raises(TypeError):
            len(foo)
    else:
        x = len(foo)
        assert x == 1
        assert not isinstance(x, mk.gui.Store)


def test_instance_method_not_decorated():
    """
    Instance methods that are not decorated should, by default, be reactive
    when the object is reactive.
    """
    foo = Foo(1)

    # Object is reactive.
    foo = foo.react()
    x = foo.add_auto_react(1)
    assert x == 2
    assert isinstance(x, mk.gui.Store)

    # Object is not reactive.
    foo = foo.no_react()
    x = foo.add_auto_react(1)
    assert x == 2
    assert not isinstance(x, mk.gui.Store)


# def test_class_method():
#     """
#     Class methods that are decorated with @reactive should always
#     be reactive by default. This is because the class does not have
#     a react flag that can be used to determine whether the method
#     should be reactive or not.
#     """
#     name = Foo.name()
#     assert isinstance(name, mk.gui.Store)
#     assert name == "Foo"


# def test_static_method():
#     """
#     Static methods that are decorated with @reactive should always
#     be reactive by default. This is because the class does not have
#     a react flag that can be used to determine whether the method
#     should be reactive or not.
#     """
#     static = Foo.static()
#     assert isinstance(static, mk.gui.Store)
#     assert static == 1
