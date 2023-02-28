from meerkat.interactive import Page, print
from meerkat.interactive.app.src.lib.component.core.numberinput import NumberInput

numberinput = NumberInput(value=1.3)

print(numberinput.value)

page = Page(component=numberinput, id="numberinput")
page.launch()
