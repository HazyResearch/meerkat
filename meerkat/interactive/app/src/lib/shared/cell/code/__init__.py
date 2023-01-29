from meerkat.interactive.app.src.lib.component.abstract import BaseComponent


class Code(BaseComponent):

    data: str
    language: str = "python"
    theme: str = "okaidia"
