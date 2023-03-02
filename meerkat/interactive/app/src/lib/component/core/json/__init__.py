from meerkat.interactive.app.src.lib.component.abstract import Component


class Json(Component):
    """Render a JSON object as a collapsible tree.

    Args:
        body (dict): The JSON object to render, as a Python dictionary.
        padding (float): Left padding applied to each level of the tree.
        classes (str): The Tailwind classes to apply to the component.
    """

    body: dict
    padding: float = 2.0
    classes: str = ""
