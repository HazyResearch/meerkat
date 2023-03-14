from typing import List

from meerkat.interactive.app.src.lib.component.abstract import Component


class MedImage(Component):
    """A component for displaying medical images.

    Args:
        data: An array of base64 encoded images.
        classes: A string of classes to apply to the component.
        show_toolbar: Whether to show the toolbar.
    """

    data: List[str]
    classes: str = ""
    show_toolbar: bool = False
