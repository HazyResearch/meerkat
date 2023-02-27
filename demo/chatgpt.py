"""Run ChatGPT with Meerkat's Chat page.

Requirements:
    - Download manifest: `pip install manifest-ml[chatgpt]`
    - Follow instructions to get your ChatGPT session key:
      https://github.com/HazyResearch/manifest

Run:
    CHATGPT_SESSION_KEY="your_session_key" python chatgpt.py
"""
import os
from datetime import datetime

from manifest import Manifest

import meerkat as mk
from meerkat.interactive.app.src.lib.component.core.chat import Chat

manifest = Manifest(
    client_name="chatgpt",
    client_connection=os.environ.get("CHATGPT_SESSION_KEY"),
)


def get_time_elapsed(time):
    # datetime.now() will be different for each cell, which may cause
    # inconsistencies in the time elapsed column.
    # TODO: Process the entire column at once.
    # Probably wont be that noticeable on a minute scale.
    time_elapsed = datetime.now() - time

    # Days
    days = time_elapsed.days
    if days > 0:
        return "{} day{} ago".format(days, "s" if days > 1 else "")

    # Minutes
    seconds = time_elapsed.seconds
    if seconds > 60:
        minutes = round(seconds / 60)
        return "{} minute{} ago".format(minutes, "s" if minutes > 1 else "")

    # Seconds
    return "{} second{} ago".format(seconds, "s" if seconds > 1 else "")


def _empty_df():
    df = mk.DataFrame(
        {
            "message": ["Hi! I'm a chatbot."],
            "sender": ["chatbot"],
            "name": ["ChatBot"],
            "send_time": [datetime.now()],
        }
    )
    df["time"] = mk.defer(df["send_time"], get_time_elapsed)
    return df


@mk.endpoint()
def on_send(df: mk.DataFrame, message: str):
    message_time = datetime.now()
    chatbot = manifest.run(message)
    new_df = mk.DataFrame(
        {
            "message": [message, chatbot],
            "sender": ["user", "chatbot"],
            "name": ["Karan", "ChatBot"],
            "send_time": [message_time, datetime.now()],
        }
    )
    new_df["time"] = mk.defer(new_df["send_time"], get_time_elapsed)
    df.set(df.append(new_df))


df = _empty_df()

chat = Chat(
    df=df,
    img_chatbot="https://placeimg.com/200/200/animals",
    img_user="https://placeimg.com/200/200/people",
    on_send=on_send.partial(df=df),
)

page = mk.gui.Page(
    component=chat,
    id="chat",
)
page.launch()
