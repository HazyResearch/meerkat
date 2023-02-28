# Endpoints

Endpoints are functions that are called in response to frontend events.

This set of pages will discuss what endpoints are, how to use them, and how to write your own.

(guide_endpoints_getting_started)=

# Getting Started with Endpoints

Endpoints in Meerkat allow users to define functions that

- can serve as API endpoints for Meerkat frontends
- can be used to modify state in response to user input or events
- can be used to launch and manage long-running processes

Meerkat builds on top of FastAPI, using a lightweight wrapper around FastAPI's `APIRouter` class to create endpoints. Most of the time, you will not need to interact with the FastAPI router directly.

## Creating Endpoints

Creating endpoints in Meerkat is easy, using the {py:class}`@mk.endpoint <meerkat.endpoint>` decorator.

```python
@mk.endpoint()
def hello_world():
    return "Hello World!"
```

Any function decorated with `@endpoint()` will automatically become an endpoint. These endpoints can be called using a special endpoint `/endpoint/{endpoint_id}/dispatch` that dispatches to all Meerkat endpoints.

By default, endpoints are `POST` requests that accept only body parameters, and no query or path parameters.

```python
import requests

response = requests.post(f"http://localhost:8000/endpoint/{hello_world.id}/dispatch")
print(response.text) # Hello World!
```

Endpoints are a particularly important building block for interactive applications in Meerkat. Let's look at a couple of examples.

## Running Endpoints in Response to Component Events

You will frequently pass endpoints to components when they are initialized. This pattern is designed to allow you to run endpoints in response to component events.

```python
@mk.endpoint()
def hello_world():
    return "Hello World!"

button = mk.Button(on_click=my_endpoint)
```

Here, the `Button` component has an `on_click` argument that takes an endpoint. When the button is clicked by a user in an interface, the `hello_world` endpoint will be called.

## What does `@endpoint` do?

Let's go over what `@endpoint` does under the hood.

1. Endpoints created with `@endpoint` are automatically added to the FastAPI docs (available at `/docs` when the FastAPI server is started).
2. These endpoints can be called using a special endpoint `/endpoint/{endpoint_id}/dispatch` that dispatches to all Meerkat endpoints.
3. By default, endpoints accept only body parameters, and no query or path parameters.
4. Only POST requests are allowed to these endpoints.

## Customizing Endpoints

To have greater control over endpoints, endpoints allow a couple of additional arguments. You can even use the underlying FastAPI router directly, although this should generally not be necessary.

Let's see the arguments that `@endpoint` provides with an example.

```python
# This endpoint is served at /hello/hello_world_v1
@mk.endpoint(prefix="/hello")
def hello_world_v1():
    return "Hello World!"

# This endpoint is served at /hello/world
@mk.endpoint(prefix="/hello", route="/world")
def hello_world_v2():
    return "Hello World!"
```

## Example: Creating a Counter with Endpoints

One of the important use cases for endpoints is to allow you to modify state in response to an event on the frontend.

To see this in more detail, let's look at the simple example of a counter.

Let's start by creating a variable to keep track of the counter value.

```python
count = Store(0)
```

Generally, you will want to use a `Store` to keep track of state in your application. `Store` objects are designed to work well with endpoints. In particular, all Meerkat objects including `Store` expose a `set` method that can be used to update them. **This method should only be used inside endpoints.**

Let's set up a couple of endpoints that increment and decrement the counter, and set the `count` variable to its new value.

```python
@endpoint
def increment(counter: Store):
    counter.set(counter + 1)

@endpoint
def decrement(counter: Store):
    counter.set(counter - 1)
```

Here, you must type annotate that `counter` will be a `Store` argument in the endpoints. This tells Meerkat that you you would like `counter` to remain a `Store` object inside the body of the endpoint function. By default, Meerkat always unwraps all `Store` objects and passes their underlying values to the endpoint.

We can now create buttons that will call these endpoints when they are clicked.

```python
increment_button = mk.gui.Button(title="Increment", on_click=increment.partial(count))
decrement_button = mk.gui.Button(title="Decrement", on_click=decrement.partial(count))
```

We use the `partial` method to create a new endpoint that is identical to the original endpoint, except that the `count` argument is set to the value of the `count` variable. This allows us to pass the endpoint to the `on_click` argument of the `Button` component.

Finally, we can create a component that displays the current value of the counter.

```python
counter = mk.gui.html.div(Text(data=count), classes="self-center text-4xl")
```

Putting it all together, we get the following application.

```python
page = mk.gui.Page(
    component=mk.gui.html.flex(
        [increment_button, decrement_button, counter]
    ),
)
page.launch()
```
