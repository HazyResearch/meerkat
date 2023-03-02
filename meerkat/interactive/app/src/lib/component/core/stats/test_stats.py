import meerkat as mk
from meerkat.interactive import Page
from meerkat.interactive.app.src.lib.component.core.stats import Stats

component = mk.gui.html.div(
    slots=[
        Stats(
            data={
                "stat_1": 11.34,
                "stat_2": 10000.3,
                "stat_3": 0.003,  # TODO: this displays as 0.00
                "stat_4": 10013123.3,
                "stat_5": 100131131231.3,
                "stat_6": 0.000005,  # TODO: this displays as 0.00
            },
        )
    ]
)

page = Page(component=component, id="stats")
page.launch()
