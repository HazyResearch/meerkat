# Interactive

One of the best ways of understanding your data is to interact with it.
This requires being able to easily and rapidly build interactive applications.
Meerkat brings interactivity to your fingertips by unifying Python programming and frontend development.

In Meerkat, interactivity is a way to observe and modify the state of your program.
More specifically, the way we interact with an interface tells us how the program should run or update state.
Modifying the state of some variable(s) can also *trigger* other states to change.

In this section, we will discuss the workhorses behind interactivity in Meerkat.

## {ref}`Reactive Functions <reactivity_getting_started>`

When we interact with any interface, changing some state can also impact how the program behaves - 
i.e. what other states should change and what operations should be re-run.
In other words, modifying the state of some variable(s) can *trigger* events and state changes.

In Meerkat, we can configure this trigger pipeline through **reactive functions**.
When the state of an input into a reactive function changes, the function will re-run.
If outputs of these functions are also inputs to other reactive functions, then those functions will also re-run.
This allows us to create a reactive chain.

## {ref}`Endpoints <guide_endpoints_getting_started>`

Interactive applications also need to communicate with the backend to tell it what to run or how to update state.

Meerkat supports this with {py:class}`endpoints <meerkat.endpoint>`.
Endpoints can serve as API endpoints for Meerkat frontends and can be used to modify state in response to user input or events.

## {ref}`Stores <guide_store_getting_started>`

An interactive application also needs some way of creating variables that capture
the state of the application. This is important to:
- keep track of application state that is going to change over time
- keep this state in sync between the frontend and Python code
- provide explicit ways to manipulate this state
- implement and debug the application in terms of this state

In Meerkat, this can be done with {py:class}`Markable <meerkat.mixins.reactifiable.MarkableMixin>` objects.
One special markable object is the {py:class}`Store <meerkat.Store>`. Stores can wrap arbitrary Python
objects to make them traceable and responsive to state changes.

## {ref}`Components <guide_components_getting_started>`

Modular applications are easier to build and reuse. But they require well-designed building blocks to be effective.

This can be done with {ref}`components <guide_components_getting_started>`, which are the main abstraction for building user interfaces in Meerkat. Users can build their own components and repurpose {ref}`built-in components <components_builtins>` for designing new interfaces.

## {ref}`Formatters <guide_formatters_getting_started>`

All interactive applications need to display data.
However, programming custom display logic for each data type can be tedious.

Meerkat provides a simple way to define custom display logic for your data types with {py:class}`formatters <meerkat.interactive.formatter.base.Formatter>`.
As a user, you can define different formatters for different data types (i.e. columns in your DataFrame) to display data in consistent and easy ways. {py:class}`Formatter groups <meerkat.interactive.formatter.base.FormatterGroup>` can define different modes for displaying data. For example, you many want to display images in different sizes (e.g. thumbnails, icons, original size, etc.).