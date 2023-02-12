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


def test_instance_method():
    y = mk.gui.Store(4)

    foo = Foo(1)
    x = foo.add(y)
    assert x == 5
    assert not isinstance(x, mk.gui.Store)
    assert isinstance(x, int)

    foo = foo.react()
    x = foo.add(y)
    assert isinstance(x, mk.gui.Store)
    assert isinstance(x, int)


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
