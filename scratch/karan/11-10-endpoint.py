import meerkat as mk
from meerkat.interactive.graph import endpoint

store = mk.gui.Store(0)


@endpoint
def increment(count: mk.gui.Store, step: int):
    count._ += step
    return count._


@mk.gui.interface_op
def next_count(count: int):
    return count + 1


nc = next_count(store)
button = mk.gui.Button(on_click=increment(store, 1))

mk.gui.start()
mk.gui.Interface(components=[button]).launch()
