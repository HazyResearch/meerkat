import meerkat as mk


@mk.endpoint()
def on_change(value):
    print("on_change", value, flush=True)
    select.labels.set([1, 2, 3, 4, 5])


select = mk.gui.core.Select(
    values=[1, 2, 3, 4, 5],
    labels=["one", "two", "three", "four", "five"],
    value=3,
    on_change=on_change,
)

select_no_labels = mk.gui.core.Select(
    values=[1, 2, 3, 4, 5],
    value=3,
    on_change=on_change,
)

component = mk.gui.html.div(
    slots=[select, select_no_labels],
)

page = mk.gui.Page(component=component, id="select")
page.launch()
