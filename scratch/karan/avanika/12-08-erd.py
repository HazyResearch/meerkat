import meerkat as mk
from meerkat.interactive.app.src.lib.component.multiselect import MultiSelect
from meerkat.interactive.app.src.lib.layouts import ColumnLayout

# 
# subprocess.call(blah)
# save to blah.png

# Make this component
# image = mk.gui.Image(
#     # put a src file in
# )

multiselect = MultiSelect(
    choices=['blah', 'blah2'],
    selected=[],
)
# multiselect would have a property called selected

@mk.gui.reactive
def foo(selected: list):
    return mk.DataFrame({'selected': selected.__wrapped__})

with mk.gui.react():
    df = foo(multiselect.selected)

table = mk.gui.Table(df=df)

mk.gui.start()
# mk.gui.Interface(component=mk.gui.ColumnLayout(components=[image, multiselect])).launch()
mk.gui.Interface(component=ColumnLayout(components=[multiselect, table])).launch()

# access multiselect.selected when something happens


@mk.gui.reactive
def my_fn(a, b, c, d):
    return a + b + c + d

my_store = mk.gui.Store(1)

button = mk.gui.Button(title="Click me!", on_click=my_store.set(2))
button2 = mk.gui.Button(title="Click me!", on_click=my_store.set(2))

with mk.gui.react():
    my_fn(my_store)