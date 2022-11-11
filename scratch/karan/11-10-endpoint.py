import meerkat as mk
from meerkat.interactive.graph import endpoint


@endpoint
def increment(count: mk.gui.Store, step: int):
    count._ += step
    return count._


store = mk.gui.Store(0)
choice = mk.gui.Choice(value=1, choices=[1, 2, 3])
button = mk.gui.Button(on_click=increment(store, choice.value._))


@mk.gui.interface_op
def next_count(count: int):
    return count + 1


nc = next_count(store)

mk.gui.start()
mk.gui.Interface(components=[choice, button]).launch()
