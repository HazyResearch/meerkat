import meerkat as mk


button_hello = mk.gui.Button("hello", on_click=mk.gui.endpoint(lambda: print("hello!")))

button_bye = mk.gui.Button("bye", on_click=mk.gui.endpoint(lambda: print("bye!")))
# tabs = mk.gui.Tabs({"hello": button_hello, "bye": button_bye})

breakpoint()
mk.gui.start(shareable=False)
mk.gui.Interface(
    components=[button_hello],
).launch()
