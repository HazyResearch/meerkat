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

    @mk.gui._reactive()
    @classmethod
    def name(cls):
        return cls.__name__

    @mk.gui._reactive()
    @staticmethod
    def static():
        return 1

    def sub(self, y):
        return self.x - y


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
