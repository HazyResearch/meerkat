import meerkat as mk
from app.src.lib.components import ExampleComponent

# Import and use the ExampleComponent
example_component = ExampleComponent(name="Meerkat")

# Launch the Meerkat GUI
mk.gui.start()
mk.gui.Interface(component=example_component).launch()
