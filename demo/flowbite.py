import meerkat as mk


@mk.gui.endpoint
def on_click():
    print("Clicked!")


button = mk.gui.flowbite.Button(
    slots="Click me",
    on_click=on_click,
)

card = mk.gui.flowbite.Card(
    slots=[
        mk.gui.Flex(
            slots=[button, "My Custom Card"],
            classes="flex-row items-center justify-between",
        )
    ],
    color="red",
    rounded=True,
    size="lg",
)

heading = mk.gui.html.h1(slots="Flowbite Demo", classes="text-4xl")

interface = mk.gui.Interface(
    component=mk.gui.RowLayout(slots=[heading, card]),
    id="flowbite",
)
interface.launch()
