from typing import Optional

from meerkat.dataframe import DataFrame
from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.endpoint import Endpoint
from meerkat.interactive.event import EventInterface


class OnSendChat(EventInterface):
    message: str


class Chat(Component):
    """A chat component.

    Args:
        df (DataFrame): The dataframe to sync with the chat. \
            It must have the following columns:
                message (str): The message to display.
                name (str): The name of the sender.
                time (str): The time the message was sent.
                sender (str): The sender of the message. \
                    Must be either "user" or "chatbot".

        imgChatbot (str): The image to display for the chatbot, as a URL.
        imgUser (str): The image to display for the user, as a URL.

        on_send: The `Endpoint` to call when a message is sent. \
            It must have the following signature:

            `(message: str)`

            with
                message (str): The message sent by the user.
    """

    # The dataframe to sync with the chat.
    df: DataFrame

    # The image to display for the chatbot.
    img_chatbot: str = "http://placekitten.com/200/300"

    # The image to display for the user.
    img_user: str = "http://placekitten.com/200/300"

    # Endpoint to call when a message is sent.
    # Endpoint should take a paramter called `message`, which is
    # the message sent by the user.
    # e.g. def on_send(message: str):
    on_send: Optional[Endpoint[OnSendChat]] = None
