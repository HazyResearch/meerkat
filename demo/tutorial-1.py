"""Write a question answering interface."""
import os

from manifest import Manifest

import meerkat as mk

manifest = Manifest(
    client_name="openai",
    client_connection="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",
)


@mk.endpoint()
def get_answer(answer_store: mk.Store, question: str):
    return answer_store.set(manifest.run(question))


textbox = mk.gui.Textbox()
output = mk.Store("")
button = mk.gui.Button(
    title="Ask OpenAI",
    on_click=get_answer.partial(answer_store=output, question=textbox.text),
)

page = mk.gui.Page(
    component=mk.gui.html.flexcol([textbox, button, mk.gui.Text(output)]),
    id="tutorial-1",
)
page.launch()
