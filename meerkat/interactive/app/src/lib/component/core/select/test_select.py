import meerkat as mk


@mk.gui.endpoint
def on_change(value):
    print("on_change", value)
    select.labels.set(["uno", "dos", "tres", "cuatro", "cinco"])


select = mk.gui.core.Select(
    values=[1, 2, 3, 4, 5],
    labels=["one", "two", "three", "four", "five"],
    value=3,
    on_change=on_change,
)

page = mk.gui.Page(component=select, id="select")
page.launch()
