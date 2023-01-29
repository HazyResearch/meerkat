from app.src.lib.components import ExampleComponent

import meerkat as mk

# Import and use the ExampleComponent
example_component = ExampleComponent(name="Meerkat")

# Launch the Meerkat GUI
# mk.gui.start() # not required for running with `mk run`
page = mk.gui.Page(component=example_component, id="example")
page.launch()
