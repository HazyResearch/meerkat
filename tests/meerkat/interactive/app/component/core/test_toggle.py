import meerkat as mk


def test_toggle_basic():
    toggle = mk.gui.Toggle()
    assert not toggle.value

    toggle.on_toggle(True)
    assert toggle.value

    toggle.on_toggle(False)
    assert not toggle.value
