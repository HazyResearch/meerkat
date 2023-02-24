(quickstart_interactive)=

# Quickstart: Interactive Apps

We'll do a quick tour of one of the main features of Meerkat - how to build interactive applications.

One of the main reasons for this functionality is because working with unstructured data frequently involves playing with it.
There's no better way to do that than through a simple, interactive application.
These applications can range from simple forms to gather user input inside Jupyter notebooks, to full-blown dashboards and web applications that are deployed to the cloud.

Make sure you have Meerkat {ref}`installed and running <install>` before you go through this quickstart.

## Components
In Meerkat, components allow us to split the user interface into independent, resuable pieces.
This functionality can range from quite simple, such as a slider that allows a user to choose a number, to quite complex, such as a full blown web application that helps a user explore their data.

Let's look at an example with a {py:class}`Slider <meerkat.interactive.core.Slider>` component. This code below creates a slider with an initial value of `2.0`.

```{margin}
Meerkat offers many components out-of-the-box. 
```



```python
import meerkat as mk

slider = mk.gui.core.Slider(value=2.0)
```

Once, we have our component, we want to render it. We can do this by passing the component to a {py:class}`Page <meerkat.interactive.Page>`, giving it an `id`, and launching it.

```python
page = mk.gui.Page(component=slider, id="slider")
page.launch()
```

If you are in a Jupyter notebook, `page.launch()` will render the component directly in the notebook. If you are using a script, Meerkat will give you a link where you can navigate to see the page.

Now you have a slider! When you move the slider, the value will automatically be updated in the Python program for you to use.

### Data Components
Meerkat also includes components that allow you to visualize and interact with DataFrames (see our documentation on DataFrames if you’re not familiar). Let’s take a look at an example.

```python
df = mk.get("imagenette")
gallery = mk.gui.Gallery(df, main_column="img")
```

The {class}`Gallery <meerkat.interactive.core.Gallery>` component takes in a `mk.DataFrame` as input and visualizes it in a paginated, interactive table. You can see a full list of our components {ref}`here <components_builtins>`.

````{margin}
```{admonition} Declarative
:class: important

One of Meerkat’s goals is to provide a consistent interface when working with DataFrame components. 

For example, we generally implement our DataFrame components so you can call them with MyComponent(df, kwarg1=value1, kwarg2=value2, ...). This is reminiscent of using seaborn's plotting library with pandas.
```
````

### Composing Components

Components are composable, so you can take multiple components and put them together to build an application.

```python
slider = mk.gui.Slider(value=2.0)
input_field = mk.gui.Input(value=2.0, dtype=int)  #FIXME
```

Let’s put these together.

```python
component = mk.gui.html.div([slider, input_field], classes="text-bold")
page = mk.gui.Page(component=component, id="compose")
page.launch()
```

We just used a div to stack up the two components and lay them out.
In fact, you can use html tags like `span`, `div`, `p` as components in Meerkat.
You can find a list of supported html components {ref}`here <components_builtins_html>`. 

```{important}
More on components

There’s a lot more to components in Meerkat that you can learn about in the [guide to components]. 

We go over other components you can use in Meerkat from the flowbite library, how to add components from any Svelte component library without writing any frontend code whatsoever, and how to write custom Meerkat components in Svelte (it’s not that hard, even if you haven’t written a line of HTML/CSS/Javascript)!
```

### Connecting Components
We might like to tie the values of the the slider and the input field, so that they stay in-sync. This can be done by simply passing slider.value to the input field.

```python
slider = mk.gui.Slider(value=2.0)
input_field = mk.gui.Input(value=slider.value, dtype=float). #FIXME
```


## Reactivity
In the app created above, moving the slider will not affect the displayed value. But, we would like to display the square of the current value - not the original. 

To do so, we can simply wrap the square method with the `mk.reactive()` decorator. 

```python
slider = mk.gui.Slider(value=2.0)

@mk.reactive()
def square(a: float) -> float:
	return a ** 2

result = square(slider.value)

page = mk.gui.Page(component=mk.gui.div([slider, mk.gui.div(result)]), id="square")
page.launch()
```

*How does this work?*

Our decorator makes the `square` function *reactive,* meaning that when one of its input parameters (e.g., `slider.value`) changes, the function reruns.

```{definition}
Definition (reactive function).
A function that reruns when any of its parameters change. 
```

Let’s be precise about how this happens.

1. The value of the slider is a floating point number, but when we check the type of `slider.value` , we’ll see it is actually a special Meerkat object called a `Store`. 
  
```python
type(slider.value)
```

Stores can wrap around arbitrary Python objects, while still exposing all of their methods and properties. In almost all cases, you can use a store as if it were unwrapped. 

```python
slider.value.is_integer()
```

By any Python object, we mean ***any***, so you could even do this:

```python
import pandas as pd
df = Store(pd.DataFrame({'a': [1, 2, 3], 'b': [1, 1, 2]}))
df.groupby(by='b')
```

```{margin}
Stores do have some gotchas with certain objects that you can read more about [here]() e.g. `Store(None) is None` will return `False`.
```

2. Passing this store to `square` tells Meerkat to watch `slider.value` and rerun `square` when it changes. Remember that this only works when `square` is decorated with `@mk.reactive()`.
    
```python
result = square(slider.value)
```

### Chaining Reactive Functions

The result of `squared` is a `Store`, so we can pass it into other reactive functions and create a chain! Let’s write a function `multiply` that takes the result of `square` and a coefficient and multiplies the two.

```python
input_slider = mk.gui.Slider(value=2.0)
coef_slider = mk.gui.Slider(value=2.0)

@mk.reactive()
def square(a: float) -> float:
	return a ** 2

@mk.reactive()
def multiply(a: float, coef: float) -> float:
	return 3 * a

intermediate = square(input_slider.value)
result = multiply(intermediate, coef_slider.value)

page = mk.gui.Page(component=mk.gui.div([slider, mk.gui.div(result)]), id="square")
page.launch()
```

Moving the `input_slider` sets off a chain reaction! 

The `square` function is rerun, which changes the value of the `intermediate` store, which in turn triggers the `multiply` function. At the end we get a new `result` that is displayed to the user.

Now, if we we instead move the `coef_slider`, *only* the `multiply` function will rerun. Meerkat will save the cost of running `square` again, since its input was not changed using the `input_slider`.

```{margin}
📌 **Return values of reactive functions.**
To help you chain reactive functions, remember that reactive functions will always wrap their return values in `Store` objects before returning them. This is handled automatically by Meerkat when you decorate a function with `@mk.reactive()`.

You can read more about this in the [guide to writing reactive functions](../guide/reactive/faq.md).
```

We’ll see that this issue shows up all the time in applications that we’ll want to build, and reactivity will help us address it.

- Say you have a DataFrame, and you create a view of that DataFrame by filtering it. When you edit the DataFrame, you will want the view that depends on the DataFrame to update automatically.
- More generally, you have a set of inputs that define some output, and if any of the inputs change, you will want the output to be updated automatically.
- Even more generally, you have a graph that has many sets of inputs and outputs, and you want that if some input is changed, all the outputs that depend directly or indirectly on that input are updated automatically.

## Endpoints

Often, you want the frontend to send a signal to run a function on the backend. For example, maybe you may want a button that increases `slider.value` by 1.

You can do this with ***endpoints***! Endpoints are functions that get run when an event occurs on the frontend - for example, when a button is clicked. Inside an endpoint, you can update a value of a store by calling `store.set(new_value)`.

Endpoints in Meerkat are created with the `@mk.endpoint()` decorator.

```python
@mk.endpoint()
def increment(value: mk.Store):
	value.set(value + 1)
```

Remember, we want to increase `slider.value` by 1.
We can use `endpoint.partial` to bind the endpoint to `slider.value`.
We can pass the partialed endpoint to the button’s `on_click`,
which tells the button to run this endpoint when the button is clicked.

```python
endpoint = increment.partial(value=slider.value)
button = mk.gui.Button(on_click=endpoint)
```

Let's put it all together.

```python
@mk.reactive()
def square(a: float) -> float:
	return a ** 2

@mk.endpoint()
def increment(value: mk.Store):
	value.set(value + 1)

slider = mk.gui.Slider(value=2.0)
result = square(input_slider.value)

endpoint = increment.partial(value=slider.value)
button = mk.gui.Button(on_click=endpoint)

page = mk.gui.Page(component=mk.gui.div([slider, button, mk.gui.div(result)]), id="square")
page.launch()
```
