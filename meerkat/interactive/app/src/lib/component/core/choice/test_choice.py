import meerkat as mk


choice = mk.gui.Choice(
    choices=["A", "B", "C"],
    value="A",
    gui_type="dropdown",
    title="Title",
)

choice_2 = mk.gui.Choice(
    choices=["A", "B", "C"],
    value="A",
    gui_type="radio",
    title="Title",
)

page = mk.gui.Page(
    component=mk.gui.html.flexcol(slots=[choice, choice_2]),
    id="choice",
)
page.launch()
