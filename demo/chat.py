"""
Build a simple chat app.
"""
import meerkat as mk
from meerkat.interactive.app.src.lib.component.chat import Chat

df = mk.DataFrame(
    {
        "message": ["hello"],
        "sender": ["chatbot"],
        "name": ["ChatBot"],
        "time": ["2 hours ago"],
    }
)

@mk.gui.endpoint
def on_send(df: mk.DataFrame, message: str):
    df.set(
        df.append(
            mk.DataFrame(
                {
                    "message": [message, "random message"],
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
