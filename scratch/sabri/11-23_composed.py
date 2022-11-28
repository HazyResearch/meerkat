import meerkat as mk


button_hello = mk.gui.Button(title="hello", on_click=mk.gui.endpoint(lambda: print("hello!")))

button_bye = mk.gui.Button(title="bye", on_click=mk.gui.endpoint(lambda: print("bye!")))
tabs = mk.gui.Tabs(tabs={"hello": button_hello, "bye": button_bye})

mk.gui.start(shareable=False)
mk.gui.Interface(
    components=[tabs],
).launch()
