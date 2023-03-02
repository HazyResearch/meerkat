# Custom Components
Meerkat has great support for building custom components in Svelte. This page will discuss how to write your own components.

```{margin}
If you're familiar with other web frameworks like React, Svelte should be easy to pick up.
```

This page assumes some familiarity with Svelte. If you're new to Svelte, you can read the [Svelte tutorial](https://svelte.dev/tutorial/basics) to get started. Svelte is not very difficult to learn even if you're new to web development.

```{margin}
The `@meerkat-ml/meerkat` [npm package](https://www.npmjs.com/package/@meerkat-ml/meerkat) has convenient utilities to interact with the Meerkat backend e.g. support for fetching data from data frames.
This is installed automatically when you create a Meerkat app, and you can import it in your Svelte components.
```

We try to make this process as easy as possible for you. At a high level, the steps are:
1. Create a new Meerkat app.
2. Write a Svelte component.
3. Write a Python class that inherits from `Component`, with attributes that match the Svelte component's props.
4. Import your Python component into your app and use it like any other component.


## Setup
To begin, create a new Meerkat app in an empty directory of your choice.

```bash
mk init
```

This will organize your directory with the following structure:
```
.
├── example.py
├── example.ipynb
├── app
│   ├── src
│   │   ├── lib
│   │   │   ├── components
│   │   │   │   ├── __init__.py
│   │   │   │   └── ExampleComponent.svelte
└── setup.py
```

```{margin}
_If this setup process fails, let us know on [Github](https://github.com/hazyresearch/meerkat/issues) or [Discord](https://discord.gg/pw8E4Q26Tq)._
```

Try running the app with `mk run example.py` and you should be able to see a page that displays the single `ExampleComponent` component.

## Writing a Custom Component
Let's start by walking through the `ExampleComponent` component and modifying it in a couple of simple ways to get a feel for the development process. We'll look at a few files in order:
- The Svelte component at `app/src/lib/components/ExampleComponent.svelte`.
- The app at `example.py`.
- The Python component at `app/src/lib/components/__init__.py`.

Before you proceed, make sure you have the Meerkat app running with `mk run example.py`, and the app open in your browser.

#### The Svelte Component
Open `app/src/lib/components/ExampleComponent.svelte`, and you'll see the following code:
```html
<!-- app/src/lib/components/ExampleComponent.svelte -->
<script lang="ts">
    export let name: string = "World";
</script>

<h1 class="text-center text-xl underline bg-purple-200">Hello {name}!</h1>
```
You'll notice that this is just a regular Svelte component that uses TypeScript. You can write your components in JavaScript or TypeScript, use any Svelte features you like, and import any npm packages you need. Generally, we recommend using Tailwind CSS for styling components that will be used with Meerkat (although you are free not to).

Since this is run in `dev` mode, you can try changing the word `Hello` to `Hi` and save the file. You should see the change reflected in your browser immediately due to hot reloading.

#### The App
Next, let's see how this component is used in the app. Open `example.py` and you'll see the following code:
```python
# example.py
from app.src.lib.components import ExampleComponent

import meerkat as mk

# Import and use the ExampleComponent
example_component = ExampleComponent(name="Meerkat")

# Launch the Meerkat GUI
page = mk.gui.Page(component=example_component, id="example")
page.launch()
```
This should look pretty similar to using Meerkat's built-in components. The only difference is that we're importing the `ExampleComponent` from `app.src.lib.components`. 

Try changing the `name` argument in `ExampleComponent` from `"Meerkat"` to something else and saving the file. You should see the change reflected in your browser automatically, although the live reloading will be a bit slower since the Python code is being re-run.

#### The Python Component
Finally, let's get familiar with the `ExampleComponent` Python class. Open `app/src/lib/components/__init__.py` and you should see the following code:
```python
# app/src/lib/components/__init__.py
from meerkat import classproperty
from meerkat.interactive import Component


class LibraryMixin:
    @classproperty
    def namespace(cls):
        return "custom"


# Component should always be the last base class
class ExampleComponent(LibraryMixin, Component):
    name: str = "World"
```
```{margin}
Use `snake_case` for attribute names in Python classes, and `camelCase` for attribute names in Svelte components e.g. `my_attribute` in Python and `myAttribute` in Svelte. Meerkat will automatically convert between the two.
```

Every Svelte component that you write should have a corresponding Python class that inherits from `Component`. The Python class should have attributes that match the props of the Svelte component, with appropriate types. You should override the `namespace` class property to a unique string that will be used to identify your component. This is used to avoid name collisions with other components in Meerkat.

The Python class must be defined in the same subdirectory as the Svelte component, as Meerkat can then automatically correspond the two.

This covers the basics of writing a custom component. Let's now look at a few more complex examples.

## Example: Creating a Counter Component
Let's create a new component called `Counter` that displays a number that can be incremented and decremented.

#### The Svelte Component
We'll start by creating a new Svelte component at `app/src/lib/components/Counter.svelte`. We'll use the following code:

```html
<!-- app/src/lib/components/Counter.svelte -->
<script>
    import { createEventDispatcher } from 'svelte';
    const dispatch = createEventDispatcher();

    export let value = 0;
    
    function onDecrement(e: Event) {
        value -= 1;
        dispatch('decrement', { count: value });
    }

    function onIncrement(e: Event) {
        value += 1;
        dispatch('increment', { count: value });
    }

</script>

<button on:click={onDecrement}>-</button>
<span>{value}</span>
<button on:click={onIncrement}>+</button>
```

This component has two buttons that increment and decrement the `value` prop, and a span that displays the current value. It also dispatches custom events when the value is incremented or decremented, with each event sending the new value as a payload.

#### The Python Component
Next, we'll create a Python class that inherits from `Component` and matches the props of the Svelte component. We'll add this to `app/src/lib/components/__init__.py`:

```{margin}
While `value` is typed as `int` in `Counter`, it will automatically be wrapped in a `Store` object when the component is initialized. This is done by `Component` as a handy convenience.

As a `Store`, it will be synchronized by Meerkat with the `value` prop in the Svelte component e.g. you can write reactive functions that depend on `value` in Python, and they will automatically re-run when the counter is incremented or decremented.
```

```python
# app/src/lib/components/__init__.py

class OnIncrementCounter(EventInterface):
    """Event sent when the counter is incremented."""
    count: int


class OnDecrementCounter(EventInterface):
    """Event sent when the counter is decremented."""
    count: int


class Counter(LibraryMixin, Component):
    """
    A simple counter component that displays a number and allows it to be 
        incremented and decremented.

    Args:
        value: The initial value of the counter.
        on_increment: An endpoint that will be called when the counter is incremented.
        on_decrement: An endpoint that will be called when the counter is decremented.
    """
    value: int = 0

    on_increment: Optional[Endpoint[OnIncrementCounter]] = None
    on_decrement: Optional[Endpoint[OnDecrementCounter]] = None
```

In addition to matching the props of the Svelte component, this class has two endpoint attributes, `on_increment` and `on_decrement`, which are called when the counter dispatches the `increment` and `decrement` events. Using event interfaces e.g. `OnIncrementCounter` allows you to clearly define the payload of the events so a user of this component can write endpoints that accept the correct payload.


#### The App

Let's see the `Counter` component in action. We'll create a new file called `counter.py` next to `example.py` and add the following code:


```python
# counter.py
from app.src.lib.components import Counter

import meerkat as mk

@mk.endpoint()
def on_increment(count: int):
    print(f"Counter incremented to {count}.")

# Import and use the Counter
counter = Counter(
    value=0,
    on_increment=on_increment,
)

# Launch the Meerkat GUI
page = mk.gui.Page(component=counter, id="counter")
page.launch()
```

And that's it!

## Components with Slots
```{margin}
_We are working to expand support for named slots in Python components._
```
Svelte components can have slots that allow you to pass in other components. Meerkat currently provides a rudimentary way to use slots in Python.

Suppose we have a component called `Card` that has a slot for content. This can look like something like this in Svelte.

```html
<!-- app/src/lib/components/Card.svelte -->
...
<div class="card">
    <slot />
</div>
```

To use this slot in Python, we can take advantage of the `Slottable` mixin. This mixin injects a `slots` attribute into the component that can be used to pass in other components. For example, we can create a `Card` component in Python as follows:

```python
# app/src/lib/components/__init__.py

class Card(LibraryMixin, Slottable, Component):
    ...
```

And then to use this,

```python
# Create a card with a Counter and Text in the slot
card = Card(slots=[Text("Counter"), counter])
```

## Components with Data Frames

Meerkat makes it easy for you to use data frames in your component props. One of the design patterns that we particularly like is to design Python components that are initialized with one or more data frames, a set of column names and other optional parameters.

For example,
```{margin}
This declarative component design pattern is particularly inspired by [Seaborn](https://seaborn.pydata.org/).
```
```
component = Component(
    df=...,
    main_column=...,
    another_column=...,
    yet_another_column=...,
    optional_param=...,
    optional_param_2=...,
)
```

Let's go over how to actually support data frames in your component.

#### The Svelte Component

Let's suppose you're designing a component that visualizes a data frame in some way. You'll need to 
pass a data frame prop to the component, and then fetch the data frame from the Python side. You can do this using the `DataFrameRef` type and the `fetchSchema` and `fetchChunk` functions, provided in the `@meerkat-ml/meerkat` npm package that is automatically installed in every Meerkat app.

```html
<script>
    import type { DataFrameRef } from "@meerkat-ml/meerkat/utils/dataframe";

    // DataFrameRef is a type that represents a reference to a data frame on the
    // Python side.
    export let df: DataFrameRef;

    // Fetch the data frame schema from the Python side.
    $: schemaPromise = fetchSchema({
		df: df,
        // ... other arguments
	}: DataFrameSchemaRequest);


    // Fetch a chunk of the data frame from the Python side.
    $: chunkPromise: DataFrameChunk = fetchChunk({
        df: df,
        start: 0,
        end: 1000,
        columns: ["column1", "column2", "column3"],
        // ... other arguments include posidxs, keyidxs, keyColumn, variants, shuffle
    }: DataFrameChunkRequest);
</script>

{#await schemaPromise then schema}
    <!-- schema is an object with keys 
            `id`, `columns`, `nrows`, `primaryKey`  
        
        `columns` is itself an array of objects with keys 
            `name`, `type`, `cellComponent`, `cellProps`, `cellDataProp`
    -->
    ...
{/await}
{#await chunkPromise then chunk}
    <!-- chunk is a DataFrameChunk object with keys 
            `columnInfos`, `columns`, `posidxs`, `keyidxs`, `rows`, `fullLength`, `primaryKey`
        and methods
            `getCell(row, column)`, `getColumn(column)`, `length()`
    -->
    ...
{/await}
```
Refer to the definition of `DataFrameSchemaRequest` and `DataFrameChunkRequest` in `@meerkat-ml/meerkat/utils/dataframe` for more details.

#### The Python Component
Once you've defined the Svelte component, you can define the Python component, which will be quite simple.

```python
def MyComponent(LibraryMixin, Component):
    df: DataFrame
    # ...
```
and then use it normally.

```python
component = MyComponent(df=df, ...)
```
