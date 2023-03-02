(components_index)=

# Components

Components are the building blocks of interactive user interfaces in Meerkat.

This set of pages will discuss the following topics:

- what components are
- how to compose components
- how to write your own components

(guide_components_getting_started)=

## Getting Started

Components are the main abstraction for building user interfaces in Meerkat.

## Creating a Component

To use a component in Meerkat, just import and initialize it. For example, the `Slider` component allows you to create a slider that can be used to select a value from a range of values.

```{margin}
`on_change` is an endpoint that will be run when the slider value changes. All endpoint component attributes have the form `on_<event>` in Meerkat.
```

```python
slider = Slider(
    value=0.25,
    min=0,
    max=1,
    step=0.1,
    on_change=on_change,
)
```

```{margin}
*If you have ideas for new components you would like to be able to use within Meerkat, let us know on [Github](https://github.com/hazyresearch/meerkat/issues) or [Discord](https://discord.gg/pw8E4Q26Tq)!*
```

Meerkat ships with a variety of components that you can use out-of-the-box, ranging from low-level HTML elements like `div` and `p`, to basic widgets like `Slider` and `Textbox`, to complex components like `Gallery`. A list of the components available in Meerkat can be found [here](inbuilts.rst).

## Understanding Components

```{margin}
The base class for all components is the `BaseComponent` class. `Component` inherits from `BaseComponent` with sensible defaults. In rare circumstances, you may need to subclass `BaseComponent` instead of `Component`.
```

Components are defined as subclasses of the `Component` class, which takes care of most of the boilerplate for you.
Let's take a look at how the `Slider` component is implemented to understand how components work in Meerkat.

```python
from typing import Any, Optional, Union

from meerkat.interactive import Component, Endpoint, EventInterface


class OnChangeSlider(EventInterface):
    value: Any


class Slider(Component):
    """A slider that allows the user to select a value from a range.

    Args:
        value: The current value of the slider.
        min: The minimum value of the slider.
        max: The maximum value of the slider.
        step: The step size of the slider.
        disabled: Whether the slider is disabled.
        classes: The Tailwind classes to apply to the component.
        on_change: An endpoint to run when the slider value changes.
    """

    value: Union[int, float] = 0
    min: Union[int, float] = 0.0
    max: Union[int, float] = 100.0
    step: Union[int, float] = 1.0
    disabled: bool = False
    classes: str = "bg-violet-50 px-4 py-1 rounded-lg"

    on_change: Optional[Endpoint[OnChangeSlider]] = None
```

Let's break this down.

**Standard attributes.** Component attributes are defined with Python type annotations, and optionally a default value. These attributes should be passed to the component constructor as keyword (and not positional) arguments. Components use Pydantic to ensure that their arguments are valid.

```python
Slider(0.25) # won't work, the Pydantic model underneath will complain
Slider(value=0.25) # works
```

**Automatic Store conversion.** At initialization, the `Component` class converts standard attributes into `Store` objects (this excludes attributes with type annotations that are other Meerkat objects like `DataFrame` or `Endpoint`), which automatically synchronizes their value with the frontend. This allows component attributes to be used as inputs to reactive functions, as well as passed to other components in order to tie attributes together.

```python
print(slider.value) # Store(0.0), not 0.0
```

```{margin}
`Endpoint` and `EndpointProperty` are only relevant if you write a custom Svelte component. Think of them as identical if you're a Python user.
```

**Endpoint attributes.** All attributes defined with `on_<event>` are endpoint attributes, and must have `Endpoint` or `EndpointProperty` as the type annotation. Endpoint attributes are run when the corresponding event is triggered on the frontend. These attributes are generally optional to pass to the constructor.

**Event interfaces.** Event interfaces are defined with `On<Event><Component>` by convention, and must subclass `EventInterface`. They document the arguments that the endpoint attributes should accept, and are used in type annotations of the form `Endpoint[On<Event><Component>]`. For example, the `on_change` endpoint of the `Slider` component is expected to be `Endpoint[OnChangeSlider]`.

```python
class OnChangeSlider(EventInterface):
    value: Any
```

This means that `value` (of any type) will be passed to the `on_change` endpoint as an argument from the frontend i.e. this endpoint must accept a `value` argument.

## Composing and Laying Out Components

A simple component like `Slider` will often not be very useful on its own. Components become much more powerful when many of them can be composed and laid out into an interface. For example, let's create a `Slider` component and pass it to a `Text` component to display the current value of the slider.

```python
slider = Slider()
text = Text(slider.value)
```

Here, you simply pass the `Store` object `slider.value` to the `Text` component. The `Text` component will automatically update whenever the value of the `slider.value` store changes.

````{margin}
If you're curious, this layout will end up looking something like this on the frontend.
```html
<div class="flex flex-col">
    <Slider ... />
    <Text ... />
</div>
````

Meerkat will do this automatically, so don't worry about it.

````

Meerkat includes a subset of `html` tags that are quite useful when you want to layout components. For example, you can use the `div` component to create a `div` element that contains the `slider` and `text` components we just created.

```python
layout = div([slider, text], classes="flex flex-col")
````

As its first argument, `div` accepts `slots`, which is a list of components that will be rendered inside the `div` element. You'll see that many other components in Meerkat have this special `slots` argument: all of them inherit from a mixin class called `Slottable` in addition to `Component`.

```{margin}
Tailwind is an incredibly simple way to style components without worrying about low-level CSS. It's easy even if you've never used CSS before. Read more about it [here](https://tailwindcss.com/).
```

Notice that you can directly pass a list of classes to the `classes` attribute of the `div` component. Whenever you see a component that accepts a `classes` attribute, you can pass a list of [Tailwind CSS](https://tailwindcss.com/) classes to the component to style it.

## Rendering Components

To actually render and view components, you must create a `Page`.

````{margin}
```{caution}
As convention, ensure that the main page of your application is assigned to a variable called `page`.
```
````

```python
page = Page(component=layout, id="my-page")
page.launch()
```

The `Page` class is the main entrypoint for creating interactive user interfaces with Meerkat. It takes a `component` argument which is the root component of the page, and an `id` argument which sets up a URL for the page.

## Running the Application

To run the application, simply put your code into a script, and run the following command in your terminal:

```bash
mk run my_app.py
```

You can also run `page.launch()` in a Jupyter notebook to view the page in the output cell.
