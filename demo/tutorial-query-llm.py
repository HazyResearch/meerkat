"""Write a question answering interface."""
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
    title="Ask an LLM",
    on_click=get_answer.partial(answer=answer, question=textbox.text),
)

page = mk.gui.Page(
    component=mk.gui.html.flexcol([
        textbox, button, mk.gui.Markdown(answer)
    ]),
    id="query-llm",
)
page.launch()
