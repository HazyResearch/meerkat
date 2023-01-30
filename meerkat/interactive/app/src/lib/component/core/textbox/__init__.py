from typing import Optional

from meerkat.interactive.app.src.lib.component.abstract import Component


class Textbox(Component):

    text: Optional[str] = ""
    title: str = ""
