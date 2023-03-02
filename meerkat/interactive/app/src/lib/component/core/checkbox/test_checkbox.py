import meerkat as mk


@mk.endpoint()
def on_change(checked: bool):
    print(checked, flush=True)


checkbox = mk.gui.core.Checkbox(
    slots=[mk.gui.html.div(slots=["Checkbox"], classes="text-purple-500")],
    checked=True,
    classes="bg-violet-50 p-2 rounded-lg w-fit",
)
checkbox.on_change = on_change.partial(checked=checkbox.checked)

page = mk.gui.Page(component=checkbox, id="checkbox")
page.launch()
