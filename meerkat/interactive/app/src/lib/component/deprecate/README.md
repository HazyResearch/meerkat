## Adding a Component
Creating a component requires defining a component in Svelte with
 corresponding Python bindings. 

1. Create a folder for the component, `<componentnameinlowercase>`.
2. Create a Svelte file for the component inside this folder, `<ComponentNameInCamelCase>.svelte`.
   1. All props must be stores in this file, and their value will be accessed using `$propname`.
   2. Use the following block of code to use functions that make backend requests.
   ```javascript
   import { getContext } from "svelte";
   { <context_that_you_need> } = getContext('Interface')
   ```
3. Add a `__init__.py` file inside.
   1. Populate this file with a Python class corresponding to the component.
    ```python
    from meerkat.interactive.graph import Pivot, Store, make_store
    from ..abstract import Component
    
    class ComponentNameInCamelCase(Component):

        def __init__(self):
            super().__init__()
            self.propname = make_store("")
            # ...more props

        @property
        def props(self):
            return {
                "propname": self.propname.config,
                # ...more props
            }
    ```
    2. Add an import for this class to `meerkat/interactive/__init__.py`.