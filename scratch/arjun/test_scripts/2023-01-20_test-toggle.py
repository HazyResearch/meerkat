import meerkat as mk


@mk.gui.endpoint
def on_toggle(value: bool):
    print("Toggle:", value)


toggle = mk.gui.Toggle(title="On Toggle", on_toggle=on_toggle)
interface = mk.gui.Interface(toggle, id="test-toggle")
mk.gui.start()
interface.launch()
