# Core Concepts in a Nutshell

<!-- #FIXME this file is random and doesn't parallel the structure of the docs -->

## Data Structures

1. Create `Store` objects to keep track of simple state variables, and keep them in sync with the frontend.
2. Create `DataFrame` objects to keep track of complex, data state.
3. Think of `Store` objects as "by-value", in that their raw value is sent to the frontend.
4. `Stores` behave like the values they wrap, so you can pretend the `Store` is just its value (with one exception: use `Store.value` inside `@endpoint` functions for clarity).
5. Think of `DataFrame` objects as "by-reference", in that a reference to the data is sent to the frontend, and the frontend can request the data when it needs it.

## Reacting to Changes

1. Use the `@reactive` decorator to register a function as reactive.
2. Use reactive functions inside the `with react()` context manager. The function will be called whenever any of its arguments change.
3. Use reactive functions as normal Python functions without the `with react()` context manager, or inside a `with no_react()` context manager.
4. Think of reactive functions as a way to create views of data. Use them to return new `Store` and `DataFrame` objects, than can then be visualized.
5. Chain reactive functions by consuming the output of one as the input to another, in order to create complex reactive applications.
6. Put simple Python statements over `Store` objects inside `with react()` in order to make them reactive.

## Creating Endpoints

1. Use the `@endpoint` decorator to register a function `fn` as an endpoint, returning an Endpoint object.
2. Use `fn.partial(...)` to create a new Endpoint object that fills in some of the arguments of `fn`.
3. Use `fn.run(...)` to run `fn` with the given arguments.
4. Pass Endpoint objects to `on_xxx` Component attributes, in order to call the endpoint when an event occurs.

## Data Manipulation

1. Manipulate `Store` and `DataFrame` objects inside `@endpoint` functions, and use the `.set()` method to update them.
2. Do not manipulate these objects inside `@reactive` functions, as this will cause an infinite loop.

## Attaching Components

1. Initialize any of the available Components by passing in `Store`, `DataFrame` and `Endpoint`, alongside other arguments.
2. Think of Components as a way to manipulate and visualize data.
3. Manipulate `Store` objects directly on the frontend via user input.
4. Manipulate `Store` and `DataFrame` objects by dispatching to an `Endpoint`.
5. Build your own Component objects by writing a single Svelte file per component.

## Gotchas

1. Using Python primitives instead of `Store` objects, and expecting them to be in sync with the frontend.
