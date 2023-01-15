
import meerkat as mk


md_component = mk.gui.Markdown(
    "# Hello world!\n"
    "- This is a list\n"
    "- This is another item\n"
)

mk.gui.start()
mk.gui.Interface(
    components=[md_component],
).launch()