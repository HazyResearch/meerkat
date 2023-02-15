import meerkat as mk


def test_basic():
    choices = ["a", "b", "c"]
    value = mk.gui.Store("")
    choice = mk.gui.Choice(choices=choices, value=value)

    choice.on_select(1)
    assert value == "b"


def test_on_select():
    other = mk.gui.Store("")

    @mk.endpoint()
    def on_select(new_value: str):
        other.set(new_value)

    choices = ["a", "b", "c"]
    value = mk.gui.Store("")
    choice = mk.gui.Choice(choices=choices, value=value, on_select=on_select)

    choice.on_select(1)
    assert value == "b"
    assert other == "b"


def test_on_select_no_param():
    other = mk.gui.Store("")

    @mk.endpoint()
    def on_select():
        other.set("set")

    choices = ["a", "b", "c"]
    value = mk.gui.Store("")
    choice = mk.gui.Choice(choices=choices, value=value, on_select=on_select)

    choice.on_select(1)
    assert value == "b"
    assert other == "set"


def test_not_string():
    choices = [1, 2, 3]
    value = mk.gui.Store(None)
    choice = mk.gui.Choice(choices=choices, value=value)

    choice.on_select(1)
    assert value == 2


# Table
# gallery
# toggle
# match
# discover


# multiselect
