# Components

Components are the building blocks of interactive user interfaces in Meerkat.

This set of pages will discuss the following topics:
- what components are
- how to compose components
- how to write your own components


## What are components?
Components are the main abstraction for building user interfaces with Meerkat.
A list of components can be found at XXXX.

For example, the `Slider` component allows you to create a slider that can be used to select a value from a range of values.

```python
slider = Slider(
    value=0.25,
    min=0,
    max=1,
    step=0.1,
    on_change=on_change,
)
```

We can take a look at how the `Slider` component is implemented to understand how basic components are built.

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
- Standard attributes of the component can be defined with Python type annotations, and optionally a default value. These attributes are required to be passed to the constructor of the component as keyword (and not positional) arguments.
```python
Slider(0.25) # fails
Slider(value=0.25) # succeeds
```
- At initialization, the `Component` class automatically converts these attributes into `Store` objects which synchronizes their value with the frontend. This allows them to be used as inputs to reactive functions, as well as passed to other components in order to tie attributes together.
```python
print(slider.value) # Store(0.0)
```
- All endpoint attributes are defined with `on_<event>`, and `Endpoint` or `EndpointProperty` as the type annotation (for component users, they are indistinguishable). Endpoint attributes are run when the corresponding event is triggered on the frontend.
- Event interfaces are defined conventionally with `On<Event>Component`, and must subclass `EventInterface`. These are used to document the arguments provided in the event payload through type annotations of the form `Endpoint[On<event>Component]`. For example, the `on_change` endpoint of the `Slider` component is defined as `Endpoint[OnChangeSlider]`. This means that a `value` attribute will be passed to the endpoint as an argument from the frontend i.e. the endpoint should be accept a `value` argument.


## Composing Components
Components can be composed together to create more complex user interfaces. For example, we can create a `Slider` component and pass it to a `Text` component to display the current value of the slider.

```python
slider = Slider()
text = Text(slider.value)
```

However, this hasn't yet addressed the problem of laying out these components. Meerkat includes a subset of `html` tags that can be used to layout components. For example, we can use the `div` tag to create a `div` element that contains the `slider` and `text` components.

```python
layout = div([slider, text], classes="flex flex-col")
```
