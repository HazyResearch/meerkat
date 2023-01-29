from typing import Optional

from ..abstract import Component


class Textbox(Component):

    text: Optional[str] = ""
    title: str = ""
