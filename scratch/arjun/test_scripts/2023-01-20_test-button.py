import meerkat as mk

button = mk.gui.Button(
    title="Click me!", on_click=mk.gui.Endpoint(lambda: print("Clicked!"))
)
interface = mk.gui.Interface(mk.gui.RowLayout(slots=[button]), id="test-button")
mk.gui.start()
interface.launch()
