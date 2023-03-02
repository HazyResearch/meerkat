import meerkat as mk
from meerkat.interactive import Page
from meerkat.interactive.app.src.lib.component.core.tabs import Tabs

tabs = Tabs(
    tabs={
        "Tab 1": mk.gui.core.Text("Tab 1"),
        "Tab 2": mk.gui.core.Text("Tab 2"),
    }
)


page = Page(component=tabs, id="tabs")
page.launch()
