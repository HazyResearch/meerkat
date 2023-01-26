from typing import Optional

from ..abstract import AutoComponent


class Textbox(AutoComponent):

    text: Optional[str] = ""
    title: str = ""
