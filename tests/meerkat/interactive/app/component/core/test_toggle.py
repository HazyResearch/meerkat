import meerkat as mk


@mk.endpoint()
def _change_value(value: mk.Store, new_value: bool):
    value.set(new_value)


def test_toggle_basic():
    toggle = mk.gui.Toggle()

    assert not toggle.value
    assert isinstance(toggle.value, mk.Store) and isinstance(toggle.value, bool)

    value: mk.Store[bool] = toggle.value
    _change_value(value, True)
    assert toggle.value

    _change_value(value, False)
    assert not toggle.value
