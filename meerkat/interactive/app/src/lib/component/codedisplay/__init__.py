from enum import Enum

from ..abstract import Component

class CodeDisplay(Component):

    data: str
    language: str = "python"
