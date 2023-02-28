---
file_format: mystnb
kernelspec:
  name: python3
---

(quickstart_interactive)=

# Quickstart: Interactive Apps

We'll do a quick tour of how to build interactive applications with Meerkat. Along with our data frames, we also provide tools for you to build applications over unstructured data. If you haven't already, go check out our {ref}`Quickstart: DataFrames <quickstart-df>` guide to learn more about the basics of working with data frames in Meerkat.

Working with unstructured data frequently involves interacting with it and visualizing it.
There's no better way to do that than through an interactive application.
These applications can range from simple user input forms inside Jupyter notebooks to full-blown dashboards and web applications that are deployed to the cloud.

Make sure you have Meerkat {ref}`installed and running <install>` before you go through this quickstart.

## 🖼️ Components: Display Elements

In Meerkat, components split up the visual, user interface into independent, resuable pieces.
Each piece can be simple, such as a slider that allows you to choose a number, or complex, such as a dashboard that helps you explore your data.

Let's look at an example with a {py:class}`meerkat.interactive.core.Slider` component. The code below creates a slider with an initial value of `2.0`.

````{margin}
```{admonition} In-Built Components and Customization
:class: tip
Meerkat offers many components out-of-the-box, and gives you a lot of support to create custom components, whether in pure Python, or with a little bit of Svelte. Check out our {ref}`Components guide <component>`.
```
````

```{code-cell} ipython3
import meerkat as mk

input_slider = mk.gui.core.Slider(value=2.0)
```
<!-- <iframe src="https://en.wikipedia.org/wiki/HTML_element#Frames" height="345px" width="100%"></iframe> -->
<iframe src="_static/gui/build/index.html" height="345px" width="100%"></iframe>

All components live under the `mk.gui.*` namespace e.g. `mk.gui.core` (core Meerkat components), `mk.gui.html` (html tags as components), `mk.gui.flowbite` (Flowbite components) and `mk.gui.plotly` (Plotly components).

Once a component is created, it needs to be rendered. We can do this by passing the component to a {py:class}`meerkat.interactive.Page`, giving it an `id`, and launching it.

```python
page = mk.gui.Page(input_slider, id="quickstart")
page.launch()
```

If you're in a Jupyter notebook, `page.launch()` will render the component directly in the notebook. If you're in a Python script, Meerkat will give you a link where you can navigate to see the page in a browser.

Now we have a slider! When we move the slider, the value will automatically be updated in the Python program for us to use.

### 📋 Data Components: Display Data Frames

Meerkat also includes components that allow us to visualize and interact with data frames (see our quickstart on data frames if you're not familiar). Let's take a look at an example.

```{code-cell} ipython3
df = mk.get("imagenette", version="160px")
gallery = mk.gui.Gallery(df, main_column="img")
```

The {class}`~meerkat.interactive.core.Gallery` component takes in a `mk.DataFrame` as input and visualizes it in a paginated, interactive table. Check out other data components like `Table`, `Filter` and `Match` in the list of available components {ref}`here <components_inbuilts>`.

````{margin}
```{admonition} Declarative Component Interfaces
:class: note

One of our goals in Meerkat is to provide a consistent interface for data frame components.

For example, we generally implement DataFrame components so they can be called with `ComponentName(df, kwarg1=value1, kwarg2=value2, ...)`. This is reminiscent of how data frames in Pandas are used with Seaborn's plotting library.
```

````

### 🧩 Composing Components for Layout

Components are composable, so you can take multiple components and put them together to build an application. For example, let's say you have two components.

```python
input_slider = mk.gui.Slider(value=2.0)
input_field = mk.gui.NumberInput(value=2.0)
```

You can put them together like so:

```python
page = mk.gui.Page(
    mk.gui.html.div([input_slider, input_field]), 
    id="quickstart",
)
page.launch()
```

Here, a `div` is used to stack up the two components and lay them out.
You can use HTML tags like `span`, `div`, `p` as components in Meerkat.
A full list of supported HTML components is available {ref}`here <components_inbuilts_html>`.

```{admonition} More on Components
:class: note
There's a lot more to Meerkat components that you can learn about in the {ref}`Components <components_index>` guide. We go over other components in Meerkat from the [flowbite library](https://flowbite.com/docs/getting-started/introduction/), how to add components from any Svelte component library using only Python, and how to write custom Meerkat components in Svelte.
```

### 🖇️ Connecting Components

We might like to tie the values of the the slider and the input, so that they stay in-sync. This can be done by simply passing `input_slider.value` to the input component.

```python
input_slider = mk.gui.Slider(value=2.0)
input_field = mk.gui.NumberInput(value=input_slider.value)
```

## 🏃‍♂️ Reactive Functions

In the app created above, moving the slider in the UI will affect the displayed value. Let's upgrade our app by displaying the square of the current value, not the original. This will require writing a function that runs every time the slider value changes. Introducing _reactive functions_!

```{admonition} Definition: _reactive function_
A function that reruns when one of its inputs changes.
```

Reactive functions in Meerkat are created with the `mk.reactive()` decorator.

Let's create a reactive function called `square`.

```{code-cell} ipython3
:tags: [remove-cell]
import meerkat as mk
```

```{code-cell} ipython3
@mk.reactive()
def square(a: float) -> float:
    return a ** 2

input_slider = mk.gui.Slider(value=2.0)

result = square(input_slider.value)
```

```python
page = mk.gui.Page(mk.gui.html.div([input_slider, mk.gui.Text(result)]), id="quickstart")
page.launch()
```

_How does this work?_

Since we invoke `square` by passing in `input_slider.value`, whenever that value changes, the function reruns.

Let's be precise about how this happens.

1. The value of the `input_slider` is a floating point number, but when we check the type of `input_slider.value`, we'll see it is actually a special Meerkat object called a `Store`.

   ```{code-cell} ipython3
   type(input_slider.value)
   ```

   A `Store` can wrap around arbitrary Python objects, while still exposing all of their functions and properties. In almost all cases, we can use a `Store` as if it were unwrapped.

   ```{code-cell} ipython3
   input_slider.value.is_integer()
   ```

   By any Python object, we mean **_any_**, so we can even do this:

   ```{code-cell} ipython3
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

### 🔗 Chaining Reactive Functions

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
    mk.gui.html.div([input_slider, coef_slider, mk.gui.Text(result)]),
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

## 🔚 Endpoints

Often, we want the frontend to trigger a function on the backend. This is exactly what _endpoints_ are for.

```{admonition} Definition: _endpoint_
A function that is run when an event occurs on the frontend.
```

Similar to reactive functions, endpoints in Meerkat are created with the `@mk.endpoint()` decorator. Inside an endpoint, you can update the value of a `Store` by calling `.set(new_value)` on it.

To demonstrate this, let's add a button that increments `slider.value` when clicked on. We'll need to define an endpoint called `increment` that takes as input a `Store`.

```{code-cell} ipython3
@mk.endpoint()
def increment(value: mk.Store):
    value.set(value + 1)
```

Next, we'll use `endpoint.partial` to bind the endpoint to `slider.value`.

<!-- TODO: explain `endpoint.partial` -->

We can create a new `Button` component and pass the partialed endpoint to its `on_click` argument, which tells the button to run this endpoint when clicked on.

```{code-cell} ipython3
button = mk.gui.Button(
    title="Increment", on_click=increment.partial(value=input_slider.value)
)
```

## 🥂 Putting it all together

```{code-cell} ipython3
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
```

```python
page = mk.gui.Page(
    component=mk.gui.html.div([input_slider, coef_slider, button, mk.gui.Text(result)]),
    id="quickstart",
)
page.launch()
```
