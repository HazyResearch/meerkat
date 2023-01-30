from meerkat.dataframe import DataFrame
from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.endpoint import Endpoint


class Chat(Component):

    # The dataframe to sync with the chat.
    df: DataFrame = None
    # The image to display for the chatbot.
    imgChatbot: str = None
    # The image to display for the user.
    imgUser: str = None

    # Endpoint to call when a message is sent.
    # Endpoint should take a paramter called `message`, which is
    # the message sent by the user.
    # e.g. def on_send(message: str):
    on_send: Endpoint = None
