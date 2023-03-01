# Tutorial 3: Basic Interactive Apps (Building A Simple Counter)

Let's go through a simple example to get familiar with interactivity. We'll build a simple counter that increments by 1 every time the user clicks on it.

## Import

First, we'll import the Meerkat library.

```python
import meerkat as mk
```

## Store

Next, let's create a `Store` object to keep track of the state of the counter. `Stores` in Meerkat are borrowed from their Svelte counterparts, and provide a way for users to create values that are synchronized between the GUI and the Python code.

```python
# Initialize the counter to 0
counter = mk.Store(0)
```

There are a couple of things to keep in mind when it comes to `Stores`.

- If the `Store` value is manipulated on the frontend, the updated value will be reflected here.
- If the `Store` value is set using the special `.set()` method here, the updated value will be reflected in the frontend.

## Endpoint

Then, we'll create an `Endpoint` in Meerkat, which is a Python function that can be called from the frontend. In this case, we'll create an endpoint that increments the counter by 1.

```python
@mk.gui.endpoint
def increment(counter: Store[int]):
    # Use the special .set() method to update the Store
    counter.set(counter + 1)
```

What's great here is that Meerkat uses FastAPI under the hood to automatically setup an API endpoint for `increment`. This can then be called by other services as well!

## Component

Next, we'll want to assemble our GUI: we want a button that we can press to increment the counter, as well as a way to display the current count value.

```python
# A button, which when clicked calls the `increment` endpoint with the `counter` argument filled in
button = mk.gui.Button(title="Increment", on_click=increment.partial(counter))
# Display the counter value
text = mk.gui.Text(data=counter)
```

Meerkat comes with a collection of prebuilt Components that can be used to assemble interfaces.

## Interface

Finally, we can start the API and frontend servers, and launch the interface.

```python
# Start the server
mk.gui.start()

# Launch the interface
mk.gui.Interface(
    # Put the components into a row layout for display
    component=mk.gui.RowLayout(
        components=[button, text]
    ),
).launch()
```
