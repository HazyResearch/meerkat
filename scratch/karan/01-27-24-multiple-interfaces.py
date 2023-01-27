import meerkat as mk

page1 = mk.gui.html.div(
    slots=[
        "Hello World!",
        mk.gui.html.a(
            href="/page2",
            slots=["Go to page 2"],
            rel="external",  # Must do this or it won't work!!!!
        ),
    ]
)

page2 = mk.gui.html.div(
    slots=[
        "Oh!",
        mk.gui.html.a(
            href="/page1",
            slots=["Go to page 1"],
            rel="external",  # Must do this or it won't work!!!!
        ),
    ]
)

# Ideally we can update the syntax so that something like this works:
# interface = mk.gui.Interface(
#     component=page1,
#     id="page1",
#     routes={
#         "/page2": page2,
#     }
# )
interface = mk.gui.Interface(
    component=page1,
    id="page1",
)
mk.gui.Interface(
    component=page2,
    id="page2",
)
interface.launch()
