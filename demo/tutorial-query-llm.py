"""Query a large language model (LLM) with a question and get an answer.

This is a tutorial on creating an `endpoint` in Meerkat that responds to
a button click.
"""
import os

from manifest import Manifest

import meerkat as mk

manifest = Manifest(
    client_name="openai",
    client_connection=os.getenv("OPENAI_API_KEY"),
    engine="text-davinci-003",
)

textbox = mk.gui.Textbox()


@mk.endpoint()
def get_answer(answer: mk.Store, question: str):
    response = manifest.run(question, max_tokens=100)
    return answer.set(response)


answer = mk.Store("")
button = mk.gui.Button(
    title="Ask an LLM 🗣️",
    on_click=get_answer.partial(answer=answer, question=textbox.text),
)

page = mk.gui.Page(
    mk.gui.html.div(
        [textbox, button, mk.gui.Markdown(answer)],
        classes="flex flex-col m-3 items-center gap-1",
    ),
    id="query-llm",
)
page.launch()
