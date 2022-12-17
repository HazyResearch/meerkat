********************************
Introduction to Interactive GUIs
********************************

Meerkat allows users to build interactive applications on top of Meerkat 
dataframes. 

One of the main reasons for this functionality is because working with unstructured data 
frequently involves playing with it. And there's no better way to do that than through
a simple, interactive application. These applications can range from simple forms to gather user input inside Jupyter notebooks,
to full-blown dashboards and web applications that are deployed to the cloud.

Underneath the hood, Meerkat relies on a FastAPI Python backend, coupled with 
Svelte for the frontend.

There are some key distinctions between Meerkat and other tools that allow users to build
interactive applications:

#. Compared to tools like `Streamlit <streamlit.io>`_, Meerkat is much more focused on supporting data and machine learning use cases. It is likely to be a better fit for users that want to bring the full power of a frontend framework like Svelte to their data science workflows.
#. Compared to tools like `Gradio <gradio.app>`_, Meerkat is much more focused on complex, reactive applications, rather than demos. It is likely to be a better fit for users that want to build full-blown applications that involve ML models, graphing and interactivity.
#. Compared to Python-to-React compilation approaches like `Pynecone <pynecone.io>`_, Meerkat is more opinionated about providing a solution for the average data science and machine learning user.

Most data science and machine learning workflows revolve around Python, and Meerkat brings the 
ability to build reactive data apps in Python to these users.

In addition, Meerkat embraces making it as easy as possible for users to write custom Svelte components if they desire. This can make it easy for users with only a passing knowledge of Javascript/HTML/CSS to build custom components without having to deal with the intricacies of the web stack.

A Simple Counter
################

Let's start with a simple example. We'll build a simple counter that increments by 1 every time the user clicks on it.


Import
------

First, we'll import the Meerkat library.

.. code-block:: python

    import meerkat as mk

Store
-----

Next, let's create a ``Store`` object to keep track of the state of the counter. ``Stores`` in Meerkat are borrowed from their Svelte counterparts, and provide 
a way for users to create values that are synchronized between the GUI and the Python code.

.. code-block:: python
    
    # Initialize the counter to 0
    counter = mk.gui.Store(0)
    
There are a couple of things to keep in mind when it comes to ``Stores``.

- If the ``Store`` value is manipulated on the frontend, the updated value will be reflected here.
- If the ``Store`` value is set using the special ``.set()`` method here, the updated value will be reflected in the frontend.
    
    
Endpoint
--------

Then, we'll create an ``Endpoint`` in Meerkat, which is a Python function that can be called from the frontend. In this case, we'll create an endpoint that increments the counter by 1.

.. code-block:: python

    @mk.gui.endpoint
    def increment(counter: Store[int]):
        # Use the special .set() method to update the Store
        counter.set(counter + 1)
        
What's great here is that Meerkat uses FastAPI under the hood to automatically setup an API endpoint for ``increment``. This can then be called by other services as well!

Component
---------

Next, we'll want to assemble our GUI: we want a button that we can press to increment the counter, as well as a way to display the current count value.

.. code-block:: python
    
    # A button, which when clicked calls the `increment` endpoint with the `counter` argument filled in
    button = mk.gui.Button(title="Increment", on_click=increment.partial(counter))
    # Display the counter value
    text = mk.gui.Text(data=counter)

Meerkat comes with a collection of prebuilt Components that can be used to assemble interfaces.

Interface
---------

Finally, we can start the API and frontend servers, and launch the interface.

.. code-block:: python
    
    # Start the server
    mk.gui.start()
    
    # Launch the interface
    mk.gui.Interface(
        # Put the components into a row layout for display
        component=mk.gui.RowLayout(
            components=[button, text]
        ),
    ).launch()
    
