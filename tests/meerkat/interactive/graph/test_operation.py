import meerkat as mk


def test_operation_with_skip():
    """Test an operation with noop conditions."""

    def skip_fn(old_x, old_y, new_x, new_y):
        # Arbitrary contrived noop function.
        return new_x == 2 or new_y == 2

    @mk.gui.react(skip_fn=skip_fn)
    def fn(x: int, y: int):
        return x + y

    @mk.gui.endpoint
    def set_xy(x: mk.gui.Store, y: mk.gui.Store, x_val, y_val):
        x.set(x_val)
        y.set(y_val)

    x = mk.gui.Store(1)
    y = mk.gui.Store(1)
    result = fn(x, y)
    assert result == 2

    set_xy(x, y, 3, 4)
    assert result == 7

    # the noop function should prevent the update
    # so the value should stay 7, not update to 5.
    set_xy(x, y, 2, 3)
    assert result == 7


def test_instance_method_with_skip():
    """Test instance method with noop conditions."""

    def skip_fn(old_y, new_y):
        # Arbitrary contrived noop function.
        return new_y == 2

    class Foo:
        def __init__(self, x: int):
            self.x = x

        @mk.gui.react(skip_fn=skip_fn)
        def fn(self, y):
            return self.x + y

    @mk.gui.endpoint
    def set_xy(y: mk.gui.Store, y_val: int):
        y.set(y_val)

    foo = Foo(1)
    y = mk.gui.Store(1)
    result = foo.fn(y)
    assert result == 2

    set_xy(y, 4)
    assert result == 5

    # the noop function should prevent the update
    # so the value should stay 7, not update to 5.
    set_xy(y, 2)
    assert result == 5
