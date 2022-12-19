Endpoints in Meerkat
--------------------

Endpoints in Meerkat allow users to define functions that 

- can serve as API endpoints for Meerkat frontends
- can be used to modify state in response to user input or events
- can be used to launch and manage long-running processes
     
Meerkat builds on top of FastAPI, with a lightweight wrapper around FastAPI's APIRouter class used to create endpoints. 

Creating endpoints in Meerkat is easy, using the ``endpoint`` decorator. 

.. code-block:: python

    from meerkat.gui import endpoint
    
    @endpoint
    def hello_world():
        return "Hello World!"
        
    mk.gui.start()
    # Visit the /docs page to see the endpoint
    
    print(hello_world) # Endpoint object
    print(hello_world.id) # ID of the endpoint
    
There are a few things to keep in mind:

1. The created endpoints are automatically added to the FastAPI docs (available at `/docs` when the FastAPI server is started). 
2. These endpoints can be called using a special endpoint `/endpoint/{endpoint_id}/dispatch` that dispatches to all Meerkat endpoints. 
3. The endpoints accept only body parameters, and no query or path parameters.
4. Only POST requests are allowed to these endpoints.

.. code-block:: python

    import requests
    
    response = requests.post("http://localhost:8000/endpoint/{endpoint_id}/dispatch")
    print(response.text) # Hello World!

To have greater control over endpoints, endpoints allow a couple of additional arguments. 
Let's see these with an example.

.. code-block:: python
    
    from meerkat.gui import endpoint
    
    @endpoint(prefix="/hello")
    def hello_world_v1():
        return "Hello World!"
    
    # This endpoint is now served at /hello/hello_world_v1
    
    @endpoint(prefix="/hello", route="/world")
    def hello_world_v2():
        return "Hello World!"
    
    # This endpoint is now served at /hello/world

    
To see why endpoints are important, let's look at the simple example of a counter.


Modifying state in response to user input.

.. code-block:: python

    import meerkat as mk
    from meerkat.interactive import endpoint
    from meerkat.interactive import Button, Div, RowLayout, Store, Text

    # Create a Store to keep track of the counter
    count = Store(0)

    # Endpoints allow you to modify state in response to user input
    @endpoint
    def increment(counter: Store[int]):
        counter.set(counter + 1)

    @endpoint
    def decrement(counter: Store[int]):
        counter.set(counter - 1)

    # Create Buttons to increment and decrement the counter
    increment_button = Button(title="Increment", on_click=increment.partial(count))
    decrement_button = Button(title="Decrement", on_click=decrement.partial(count))
    counter = Div(component=Text(data=count), classes="self-center text-4xl")

    mk.gui.start()
    mk.gui.Interface(
        component=RowLayout(
            components=[increment_button, decrement_button, counter]
        ),
    ).launch()

There are several recommendations to keep in mind when creating endpoints:

1. Always use type annotations to specify the types of the arguments to the endpoint. Meerkat uses Pydantic to validate the arguments to the endpoint.
2. Make sure to annotate `Store` and `DataFrame` arguments correctly. If you update these arguments inside an endpoint, Meerkat will automatically ensure that the changes will be reflected in the frontend.
