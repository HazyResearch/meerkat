import time
import meerkat as mk
from meerkat.interactive import Store
from meerkat.interactive.interface import Interface

a = Store(1)
b = Store(2)


@mk.gui.react()
def foo(a, b):
    time.sleep(1)
    return a + b

@mk.gui.react()
def bar(a, c):
    time.sleep(1)
    return a + c

@mk.gui.react()
def cub(c):
    time.sleep(1)
    return c

@mk.gui.endpoint()
def set_a(new_a: int):
    a.set(new_a)


@mk.gui.endpoint()
def set_b(new_b: int):
    b.set(new_b)


c = foo(a, b)
d = bar(a, c)
e = cub(c)


layout = mk.gui.html.div(
    slots=[
        mk.gui.flowbite.Button(
            on_click=set_a.partial(new_a=3),
            slots=["Set A"],
        ),
        mk.gui.flowbite.Button(
            on_click=set_b.partial(new_b=4),
            slots=["Set B"],
        ),
    ]
)

interface = Interface(component=layout, id="progress")
interface.launch()
