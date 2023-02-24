---
file_format: mystnb
kernelspec:
  name: python3
---

# Tutorial 3: Query a Large Language Model

In this tutorial, we will build a simple demo interface to ask a large language model (LLM) questions.

Through this tutorial, you will learn about:
- the concept of **endpoints**
- how to pass **endpoints** to components to run on events
- Using `Textbox` and `Button` components
- Advanced: Using [`manifest`](https://github.com/hazyresearch/manifest), a server-client manager for running LLMs
****
````{admonition} Prerequisites
:class: tip
You will need to install `manifest` and set up your OPENAI API key:

```bash
# Install manifest
pip install manifest-ml

# Set up your OpenAI key.
export OPENAI_API_KEY="your-api-key"
```
````

To get started, run the tutorial demo script

```{code-block} bash
mk demo tutorial-qa
```

You should see the tutorial app when you open the link in your browser. Let's break down the code in the demo script.

## Overview
In this interface, we want to ask the LLM for an answer to our question when we press a button.

This requires having

1. An LLM to return answers to our questions
1. An input text field to ask our question
1. A Python API that runs the LLM
1. A button to call the Python API on click

## (Aside) Setting up LLMs with Manifest
[Manifest](https://github.com/hazyresearch/manifest) is a Python library that provides a convenient interface to call foundation model APIs, e.g. large language models from OpenAI, Cohere, Together, etc.

Let's set up manifest to connect to OpenAI.


```{code-cell} ipython3
:tags: [remove-cell]
import meerkat as mk
import rich
```

```python
manifest = Manifest(
  client_name="openai",
  client_connection=os.getenv("OPENAI_API_KEY"),
  engine="text-davinci-003",
)
```
Running a prompt through manifest will be easy:

```python
manifest.run("What is the meaning of life?", max_tokens=100)
```
That's it! You can refer to the Manifest codebase to learn more about how to use it. Let's now move on to the rest of the application.

## Adding an Input TextField Component

Let's add a text field where a user can type out their questions.
Meerkat provides a `Textbox` component that we can use.

```{code-cell} ipython3
textbox = mk.gui.Textbox()
```

This component will do a few things:
- create a text field on the frontend that the user can type into
- synchronize the value of this text field with a `Store` object on the Python backend

Here, `textbox.text` will be a `Store`, a Meerkat data object that is synchronized with the value of the text field on the frontend. This will be important when we want to pass the value of the text field to our LLM API.

```{code-cell} ipython3
:tags: [remove-input]
rich.print(
  "[blue]type(textbox.text):[/blue]",
  f"\n\t{type(textbox.text)}",
)
```


## Endpoints
Now that we have our LLM and our input field, we want to write a Python function that will run the LLM on the input.

This function has to be special in a few ways:
1. It should be able to take in the value of the input field
2. It should run when an event on the frontend happens, e.g. a button is clicked

To do this, we can use the {py:class}`@mk.endpoint <meerkat.endpoint>` decorator. This decorator will convert a Python function into an `Endpoint` object, which is a special type of Python function that can be passed to components to run on events.

```{code-cell} ipython3
:tags: [remove-cell]
class DummyManifest:
    def run(self, question):
        return "42"

manifest = DummyManifest()
```


```{code-cell} ipython3
@mk.endpoint()
def get_answer(question: str, answer: mk.Store):
    response = manifest.run(question, max_tokens=100)
    answer.set(response)
    return answer
```

If you print this endpoint out, you'll see that it the decorator has converted the `get_answer` function into an {py:class}`Endpoint <meerkat.interactive.Endpoint>` object.

```{code-cell} ipython3
:tags: [remove-input]
rich.print(
  "[blue]get_answer:[/blue]",
  f"\n\t{get_answer}",
)
```

Notice that in addition to the `question`, we also pass in an `answer` argument. This allows us to update the `answer` store with the response from the LLM. When the store is updated, it will tell all frontend components that depend on this store to also update their values.

## Running Endpoints with Component Events

Now that we have our endpoint, we want to run it when a button is clicked.

We can do this by passing the endpoint to the `on_click` argument of a `Button` component.

```{code-cell} ipython3
# The store to keep track of the LLM answer.
answer = mk.Store("")

button = mk.gui.Button(
  title="Ask",
  on_click=get_answer.partial(question=textbox.text, answer=answer),
)
```

When the button is clicked, the `get_answer` endpoint will be called with the value of the `textbox.text` store as the `question` argument.

We can simulate this button click by running the `on_click` endpoint directly. Since it's an `Endpoint` object, we can call it with the `.run()` method.

```{code-cell} ipython3
:tags: [remove-output]
button.on_click.run()
```
```{code-cell} ipython3
:tags: [remove-input]
rich.print(button.on_click.run())
```

Different components expose different events that can be tied to endpoints. All endpoints start with `on_` and are followed by the name of the event. For example, the `on_click` event is tied to the `click` event of a `Button` component.

## Putting it all together
Now we display these different components on the frontend.

```{code-cell} ipython3
page = mk.gui.Page(
    component=mk.gui.html.flexcol([textbox, button, mk.gui.Text(answer)]),
    id="tutorial-query-llm",
)
```

When you run the app, you should see the following interface:

```python
page.launch()
```