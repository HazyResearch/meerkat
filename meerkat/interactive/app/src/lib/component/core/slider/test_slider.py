import meerkat as mk


@mk.endpoint()
def on_change(value):
    print("on_change", value)


slider = mk.gui.core.Slider(
    min=-2.0,
    max=2.0,
    step=0.01,
    on_change=on_change,
)

page = mk.gui.Page(component=slider, id="slider")
page.launch()
