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
    title="Increment",
    on_click=increment.partial(value=input_slider.value),
)

page = mk.gui.Page(
    component=mk.gui.html.div(
        [
            input_slider,
            coef_slider,
            button,
            mk.gui.Text(result),
        ],
        classes="flex flex-col gap-4 items-center",
    ),
    id="quickstart",
)
page.launch()
