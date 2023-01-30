import meerkat as mk


def test_on_click():
    store = mk.gui.Store("")
    button = mk.gui.Button(title="test", on_click=lambda: store.set("clicked"))

    button.on_click()
    assert store == "clicked"
