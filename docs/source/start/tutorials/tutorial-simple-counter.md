---
file_format: mystnb
kernelspec:
  name: python3
---

# Tutorial 3: Basic Interactive Apps (Building A Simple Counter)

Let's go through a simple example to get familiar with interactivity. We'll build a simple counter that increments by 1 every time the user clicks on it.



## 🔮 Importing Meerkat

First, we'll import the Meerkat library.

```{code-cell} ipython3
import meerkat as mk
```

## 🧺 Store: Keeping Track of State

Next, let's create a `Store` object to keep track of the state of the counter. `Stores` in Meerkat are borrowed from their Svelte counterparts, and provide a way for users to create values that are synchronized between the GUI and the Python code.

```{code-cell} ipython3
# Initialize the counter to 0
counter = mk.Store(0)
```

There are a couple of things to keep in mind when it comes to `Stores`.

- If the `Store` value is manipulated on the frontend, the updated value will be reflected here.
- If the `Store` value is set using the special `.set()` method here, the updated value will be reflected in the frontend.

## 🔚 Endpoint: Updating the State

Then, we'll create an `Endpoint` in Meerkat, which is a Python function that can be called from the frontend. In this case, we'll create an endpoint that increments the counter by 1.

```{code-cell} ipython3
@mk.gui.endpoint
def increment(counter: mk.Store[int]):
    # Use the special .set() method to update the Store
    counter.set(counter + 1)
```

What's great here is that Meerkat uses FastAPI under the hood to automatically setup an API endpoint for `increment`. This can then be called by other services as well!

## 🖼️ Component: Assembling the GUI

Next, we'll want to assemble our GUI: we want a button that we can press to increment the counter, as well as a way to display the current count value.

```{code-cell} ipython3
# A button, which when clicked calls the `increment` endpoint with the `counter` argument filled in
button = mk.gui.Button(title="Increment", on_click=increment.partial(counter))
# Display the counter value
text = mk.gui.Text(data=counter)
```

Meerkat comes with a collection of prebuilt Components that can be used to assemble interfaces.

## 📃 Page: Launching the GUI

Finally, we can launch the interface.

```python
# Launch the interface
mk.gui.Page(
    # Put the components into a layout for display
    mk.gui.html.div([button, text]),
    id="counter",
).launch()
```
