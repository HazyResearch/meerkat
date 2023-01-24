"""
Run ChatGPT with Meerkat's Chat interface.

Requirements:
    - Download manifest: `pip install manifest-ml[chatgpt]`
    - Follow instructions to get your ChatGPT session key:
      https://github.com/HazyResearch/manifest

Run:
    CHATGPT_SESSION_KEY="your_session_key" python chatgpt.py
"""
import os

from manifest import Manifest

import meerkat as mk
from meerkat.interactive.app.src.lib.component.chat import Chat

manifest = Manifest(
    client_name="chatgpt",
    client_connection=os.environ.get("CHATGPT_SESSION_KEY"),
)

df = mk.DataFrame(
    {
        "message": ["hello. starting chat gpt"],
        "sender": ["chatbot"],
        "name": ["ChatBot"],
        "time": ["2 hours ago"],
    }
)


@mk.gui.endpoint
def on_send(df: mk.DataFrame, message: str):
    chatbot = manifest.run(message)
    df.set(
        df.append(
            mk.DataFrame(
                {
                    "message": [message, chatbot],
                    "sender": ["user", "chatbot"],
                    "name": ["Karan", "ChatBot"],
                    "time": ["1 hour ago", "1 hour ago"],
                }
            )
        )
    )


chat = Chat(
    df=df,
    imgChatbot="https://placeimg.com/200/200/animals",
    imgUser="https://placeimg.com/200/200/people",
    on_send=on_send.partial(df=df),
)

interface = mk.gui.Interface(
    component=chat,
    id="chat",
)
interface.launch()
