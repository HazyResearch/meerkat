(quickstart_interactive)=

# Quickstart: Interactive Apps

We'll do a quick tour of one of the main features of Meerkat - how to build interactive applications.

The motivation for this functionality is that working with unstructured data frequently involves playing with it.
There's no better way to do that than through an interactive application.
These applications can range from simple user input forms inside Jupyter notebooks to full-blown dashboards and web applications that are deployed to the cloud.

Make sure you have Meerkat {ref}`installed and running <install>` before you go through this quickstart.

## Components

In Meerkat, components allow us to split the user interface into independent, resuable pieces.
This functionality can range from quite simple, such as a slider that allows a user to choose a number, to quite complex, such as a full blown web application that helps a user explore their data.

Let's look at an example with a {class}`~meerkat.interactive.core.Slider` component. This code below creates a slider with an initial value of `2.0`.

````{margin}
```{admonition}
Meerkat offers many components out-of-the-box.
```
````

```python
import meerkat as mk

input_slider = mk.gui.core.Slider(value=2.0)
```

Once, we have our component, we want to render it. We can do this by passing the component to a {class}`~meerkat.interactive.Page`, giving it an `id`, and launching it.

```python
page = mk.gui.Page(component=input_slider, id="quickstart")
page.launch()
```

If you are in a Jupyter notebook, `page.launch()` will render the component directly in the notebook. If you are using a script, Meerkat will give you a link where you can navigate to see the page in a browser.

Now we have a slider! When we move the slider, the value will automatically be updated in the Python program for us to use.

### Data Components

Meerkat also includes components that allow us to visualize and interact with DataFrames (see our documentation on DataFrames if you're not familiar). Let's take a look at an example.

```python
df = mk.get("imagenette")
gallery = mk.gui.Gallery(df, main_column="img")
```

The {class}`~meerkat.interactive.core.Gallery` component takes in a `mk.DataFrame` as input and visualizes it in a paginated, interactive table. A full list of our components is available {ref}`here <components_builtins>`.

````{margin}
```{admonition} Declarative
:class: important

One of Meerkat's goals is to provide a consistent interface when working with DataFrame components.

For example, we generally implement our DataFrame components so they can be called with `MyComponent(df, kwarg1=value1, kwarg2=value2, ...)`. This is reminiscent of how DataFrames in Pandas can be used with Seaborn's plotting library.
```

````

### Composing Components

Components are composable, so we can take multiple components and put them together to build an application. For example, let's say we have the following two components.

```python
input_slider = mk.gui.Slider(value=2.0)
input_field = mk.gui.Input(value=2.0, dtype=float) #FIXME
```

We can put them together like so:

```python
page = mk.gui.Page(
    component=mk.gui.html.div([input_slider, input_field]), id="quickstart"
)
page.launch()
```

We just used a `div` to stack up the two components and lay them out.
In fact, we can use HTML tags like `span`, `div`, `p` as components in Meerkat.
A full list of supported HTML components is available {ref}`here <components_builtins_html>`.

```{important}
**More on components.** There's a lot more to components in Meerkat that you can learn about in the {ref}`Components <components_index>` guide. We go over other components in Meerkat from the [flowbite library](https://flowbite.com/docs/getting-started/introduction/), how to add components from any Svelte component library without writing any frontend code whatsoever, and how to write custom Meerkat components in Svelte.
```

### Connecting Components

We might like to tie the values of the the slider and the input, so that they stay in-sync. This can be done by simply passing `input_slider.value` to the input component.

```python
input_slider = mk.gui.Slider(value=2.0)
input_field = mk.gui.Input(value=input_slider.value, dtype=float) #FIXME
```

## Reactive Functions

In the app created above, moving the slider in the UI will affect the displayed value. Let's upgrade our app by displaying the square of the current value, not the original. This will require writing a function that runs every time the slider value changes. Introducing _reactive functions_!

```{admonition} Definition - _reactive function_
A function that reruns when one of its inputs changes.
```

Reactive functions in Meerkat are created with the `mk.reactive()` decorator.

Let's create a reactive function called `square`.

```python
@mk.reactive()
def square(a: float) -> float:
    return a ** 2

input_slider = mk.gui.Slider(value=2.0)

result = square(input_slider.value)

page = mk.gui.Page(component=mk.gui.html.div([input_slider, mk.gui.Text(result)]), id="quickstart")
page.launch()
```

_How does this work?_

Since we invoke `square` by passing in `input_slider.value`, whenever that value changes, the function reruns.

Let's be precise about how this happens.

1. The value of the `input_slider` is a floating point number, but when we check the type of `input_slider.value`, we'll see it is actually a special Meerkat object called a `Store`.

   ```python
   type(input_slider.value) #FIXME runable
   ```

   A `Store` can wrap around arbitrary Python objects, while still exposing all of their functions and properties. In almost all cases, we can use a `Store` as if it were unwrapped.

   ```python
   input_slider.value.is_integer() #FIXME runable
   ```

   By any Python object, we mean **_any_**, so we can even do this:

   ```python
   #FIXME runable
   import pandas as pd


   df = mk.Store(pd.DataFrame({'a': [1, 2, 3], 'b': [1, 1, 2]}))
   df.groupby(by='b')
   ```

```{margin}
The `Store` object does have some gotchas with certain objects that you can read more about {ref}`here <guide_interactive_concepts_store_gotchas>` (e.g., `Store(None) is None` will return `False`).
```

2. Passing this `Store` to `square` tells Meerkat to watch `input_slider.value` and rerun `square` when it changes. Remember that this only works because `square` is decorated with `@mk.reactive()`.

   ```python
   squared_value = square(input_slider.value)
   ```

### Chaining Reactive Functions

Because it is a reactive function, the result of `square` is a `Store`, so we can pass it into other reactive functions and create a chain! Let's write a function `multiply` that takes as input a coefficient and the result of `square` and returns the product of the two.

```python
@mk.reactive()
def square(a: float) -> float:
    return a ** 2


@mk.reactive()
def multiply(coef: float, a: float) -> float:
    return coef * a


input_slider = mk.gui.Slider(value=2.0)
coef_slider = mk.gui.Slider(value=2.0)

squared_value = square(input_slider.value)
result = multiply(coef_slider.value, squared_value)

page = mk.gui.Page(
    component=mk.gui.html.div([input_slider, coef_slider, mk.gui.Text(result)]),
    id="quickstart",
)
page.launch()
```

Moving the `input_slider` sets off a chain reaction!

The `square` function is rerun, which changes the value of the `squared_value` `Store`, which in turn triggers the `multiply` function. At the end we get a new `result` that is displayed to the user.

Now, if we we instead move the `coef_slider`, _only_ the `multiply` function will rerun. Meerkat will save the cost of running `square` again, since its input (`input_slider.value`) was not changed.

```{margin}
📌 **Return values of reactive functions.**
An important concept to remember when chaining reactive functions is that reactive functions will always wrap their return values in `Store` objects before returning them. This is handled automatically by Meerkat on any function decorated with `@mk.reactive()`.

You can read more about this in the {ref}`Reactive FAQs <reactivity_faqs>`.
```

The paradigm of reactivity shows up all the time in our applications, so it is important to understand how it works. Here are some examples of when you might want to use reactivity:

- Say you have a DataFrame, and you create a view of that DataFrame by filtering it. When you edit the DataFrame, you will want the view to update automatically.
- More generally, say you have a set of inputs that define some output. If any of the inputs change, you will want the output to update automatically.
- Even more generally, say you have a graph that has many sets of inputs and outputs. If some input is changed, you will want all the outputs that depend directly or indirectly on that input to update automatically.

## Endpoints

Often, we want the frontend to trigger a function on the backend. This is exactly what _endpoints_ are for.

```{admonition} Definition - _endpoint_
A function that is run when an event occurs on the frontend.
```

Similar to reactive functions, endpoitns in Meerkat are created with the `@mk.endpoint()` decorator. Inside an endpoint, you can update a value of a `Store` by calling `.set(new_value)` on it.

To demonstrate this, let's add a button that increments `slider.value` when clicked on. We'll need to define an endpoint called `increment` that takes as input a `Store`.

```python
@mk.endpoint()
def increment(value: mk.Store):
    value.set(value + 1)
```

Next, we'll use `endpoint.partial` to bind the endpoint to `slider.value`.

<!-- TODO: explain `endpoint.partial` -->

We can create a new `Button` component and pass the partialed endpoint to its `on_click` argument, which tells the button to run this endpoint when clicked on.

```python
button = mk.gui.Button(
    title="Increment", on_click=increment.partial(value=input_slider.value)
)
```

## Putting it all together

```python
import meerkat as mk


@mk.reactive()
def square(a: float) -> float:
    return a**2


@mk.reactive()
def multiply(coef: float, a: float) -> float:
    return coef * a


@mk.endpoint()
def increment(value: mk.Store):
    value.set(value + 1)


input_slider = mk.gui.Slider(value=2.0)
coef_slider = mk.gui.Slider(value=2.0)

squared_value = square(input_slider.value)
result = multiply(coef_slider.value, squared_value)

button = mk.gui.Button(
    title="Increment", on_click=increment.partial(value=input_slider.value)
)

page = mk.gui.Page(
    component=mk.gui.html.div([input_slider, coef_slider, button, mk.gui.Text(result)]),
    id="quickstart",
)
page.launch()
```
