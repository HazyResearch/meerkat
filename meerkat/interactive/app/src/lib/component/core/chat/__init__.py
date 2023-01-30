from meerkat.dataframe import DataFrame
from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.endpoint import Endpoint


class Chat(Component):

    df: DataFrame = None
    imgChatbot: str = None
    imgUser: str = None

    on_send: Endpoint = None