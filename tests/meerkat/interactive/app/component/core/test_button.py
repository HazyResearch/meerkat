import meerkat as mk


def test_on_click():
    store = mk.Store("")
    button = mk.gui.Button(
        title="test", on_click=mk.endpoint(lambda: store.set("clicked"))
    )

    button.on_click()
    assert store == "clicked"
