from meerkat.interactive import Page, print
from meerkat.interactive.app.src.lib.component.core.textbox import Textbox

textbox = Textbox()

print(textbox.text)

page = Page(component=textbox, id="textbox")
page.launch()
