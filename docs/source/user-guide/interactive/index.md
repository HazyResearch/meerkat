# Interactive

One of the best ways of understanding your data is to interact with it.
This requires being able to easily and rapidly build interactive applications. These applications can range from simple forms to gather user input inside Jupyter notebooks, to full-blown dashboards and web applications that are deployed to the cloud. Meerkat brings interactivity to your fingertips by unifying Python programming and frontend development.

In Meerkat, interactivity is a way to observe and modify the state of your program. More specifically, the way we interact with an interface tells us how the program should run or update state. Modifying the state of some variable(s) can also _trigger_ other states to change.

Underneath the hood, Meerkat relies on a FastAPI Python backend, coupled with Svelte for the frontend.

There are some key distinctions between Meerkat and other tools that allow users to build interactive applications:

- Compared to tools like [Streamlit](streamlit.io), Meerkat is much more focused on supporting data and machine learning use cases. It is likely to be a better fit for users that want to bring the full power of a frontend framework like Svelte to their data science workflows.
- Compared to tools like [Gradio](gradio.app), Meerkat is much more focused on complex, reactive applications, rather than demos. It is likely to be a better fit for users that want to build full-blown applications that involve ML models, graphing and interactivity.
- Compared to Python-to-React compilation approaches like [Pynecone](pynecone.io), Meerkat is more opinionated about providing a solution for the average data science and machine learning user.

Most data science and machine learning workflows revolve around Python, and Meerkat brings the ability to build reactive data apps in Python to these users.

In addition, Meerkat embraces making it as easy as possible for users to write custom Svelte components if they desire. This can make it easy for users with only a passing knowledge of Javascript/HTML/CSS to build custom components without having to deal with the intricacies of the web stack.

In this section, we will discuss the workhorses behind interactivity in Meerkat.

## {ref}`Reactive Functions <reactivity_getting_started>`

When we interact with any interface, changing some state can also impact how the program behaves -
i.e. what other states should change and what operations should be re-run.
In other words, modifying the state of some variable(s) can _trigger_ events and state changes.

In Meerkat, we can configure this trigger pipeline through **reactive functions**.
When the state of an input into a reactive function changes, the function will re-run.
If outputs of these functions are also inputs to other reactive functions, then those functions will also re-run.
This allows us to create a reactive chain.

## {ref}`Endpoints <guide_endpoints_getting_started>`

Interactive applications also need to communicate with the backend to tell it what to run or how to update state.

Meerkat supports this with {class}`~meerkat.endpoint`s.
Endpoints can serve as API endpoints for Meerkat frontends and can be used to modify state in response to user input or events.

## {ref}`Stores <guide_store_getting_started>`

An interactive application also needs some way of creating variables that capture
the state of the application. This is important to:

- keep track of application state that is going to change over time
- keep this state in sync between the frontend and Python code
- provide explicit ways to manipulate this state
- implement and debug the application in terms of this state

In Meerkat, this can be done with {py:class}`Markable <meerkat.mixins.reactifiable.MarkableMixin>` objects.
One special markable object is the {class}`~meerkat.Store`. Stores can wrap arbitrary Python
objects to make them traceable and responsive to state changes.

## {ref}`Components <guide_components_getting_started>`

Modular applications are easier to build and reuse. But they require well-designed building blocks to be effective.

This can be done with {ref}`components <guide_components_getting_started>`, which are the main abstraction for building user interfaces in Meerkat. Users can build their own components and repurpose {ref}`built-in components <components_builtins>` for designing new interfaces.

## {ref}`Formatters <guide_formatters_getting_started>`

All interactive applications need to display data.
However, programming custom display logic for each data type can be tedious.

Meerkat provides a simple way to define custom display logic for your data types with {py:class}`formatters <meerkat.interactive.formatter.base.Formatter>`.
As a user, you can define different formatters for different data types (i.e. columns in your DataFrame) to display data in consistent and easy ways. {py:class}`Formatter groups <meerkat.interactive.formatter.base.FormatterGroup>` can define different modes for displaying data. For example, you many want to display images in different sizes (e.g. thumbnails, icons, original size, etc.).
