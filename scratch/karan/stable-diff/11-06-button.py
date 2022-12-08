import meerkat as mk

button = mk.gui.Button("Click me!")
other_button = mk.gui.Button("Click me too!")
textbox = mk.gui.Textbox("Enter text here")

@mk.gui.reactive(first_call=0)
def click(
    button: dict,
    text: str,
):
    """This function will trigger when either the button is clicked or the text is changed."""
    print(button)
    print(text)
    print("Clicked!")
    return button


output = click(button.value, textbox.text)

@mk.gui.reactive
def click_v2(
    button: dict,
):
    """Want a function that only triggers when the button is clicked."""
    print(button)
    # Problem: have to manually unwrap the textbox value
    print(textbox.text.value)
    print("Clicked!")
    return button


output = click_v2(button.value)

@mk.gui.reactive(on=button.value)
def click_v3(
    button: dict,
    text: str,
):
    """Triggers only when the button is clicked."""
    print(button)
    print(text)
    print("Clicked!")
    return button

output = click_v3(button.value, textbox.text)

@mk.gui.reactive(also_on=other_button.value)
def click_v4(
    button: dict,
    text: str,
):
    """Triggers when either button is clicked or if the textbox changes."""
    print(button)
    print(text)
    print("Clicked!")
    return button

output = click_v4(button.value, textbox.text)

@mk.gui.reactive(on=['button', other_button.value])
def click_v5(
    button: dict,
    text: str,
):
    """Triggers when either button is clicked or if the textbox changes."""
    print(button)
    print(text)
    print("Clicked!")
    return button

output = click_v5(button.value, textbox.text)
breakpoint()

mk.gui.start()
mk.gui.Interface(components=[button, other_button, textbox]).launch()
