"""

This script tests the following features of the choice component:
    - GUI type: Value can be either 'dropdown' or 'radio'
    - on_select: This function can either have 1 input parameters or 0 input parameters.
    - multiple types: choices can be of arbitrary types (int, str, etc.)
"""
import meerkat as mk


@mk.gui.endpoint
def on_select_dropdown(value):
    print("Selected dropdown:", value)


def on_select_radio():
    print(
        "Selected radio:", choice_radio.value, "type: ", type(choice_radio.value.value)
    )


choices = mk.gui.Store(["a", "b", "c", "d"])
value = ""
choice_dropdown = mk.gui.Choice(
    choices=choices, value=value, title="Choose a letter", on_select=on_select_dropdown
)

choices = [1, 2, 3, 4, 5, 6]
value = 0
choice_radio = mk.gui.Choice(
    choices=choices,
    value=value,
    title="Choose a number",
    on_select=on_select_radio,
    gui_type="radio",
)
interface = mk.gui.Interface(
    mk.gui.RowLayout(slots=[choice_dropdown, choice_radio]), id="test-choice"
)

mk.gui.start()
interface.launch()
