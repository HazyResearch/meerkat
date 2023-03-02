import meerkat as mk


@mk.endpoint()
def on_change(index: int):
    print("on_change", index)


radios = mk.gui.html.flex(
    slots=[
        mk.gui.core.Radio(
            name="radio",
            value="radio" + str(i),
            disabled=i == 2,
            slots=[mk.gui.html.div(slots=[f"Radio {i}"], classes="text-purple-500")],
            on_change=on_change.partial(index=i),
        )
        for i in range(1, 4)
    ],
    classes="bg-violet-50 p-0 rounded-lg w-fit text-center",
)


@mk.endpoint()
def on_change(index: int):
    print("on_change", index)


radio_group = mk.gui.core.RadioGroup(
    values=["Radio 1", "Radio 2", "Radio 3"],
    disabled=False,
    horizontal=True,
    on_change=on_change,
)

component = mk.gui.html.div(
    slots=[radios, radio_group], classes="flex flex-col space-y-4"
)

page = mk.gui.Page(component=component, id="radio")
page.launch()
