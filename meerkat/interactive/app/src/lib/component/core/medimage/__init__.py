from typing import Any, List, Optional

from meerkat.dataframe import DataFrame
from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.endpoint import EndpointProperty
from meerkat.interactive.event import EventInterface


class OnFetchInterface(EventInterface):
    """The interface for the get_data endpoint."""

    df: DataFrame
    column: str
    index: int
    dim: Optional[int] = None
    type: Optional[str] = None


class MedicalImage(Component):
    """A component for displaying medical images.

    Args:
        data: An array of base64 encoded images.
        classes: A string of classes to apply to the component.
        show_toolbar: Whether to show the toolbar.
        on_fetch: An endpoint to call when the component needs to fetch data.
    """

    data: List[str]
    classes: str = ""
    show_toolbar: bool = False
    dim: int
    segmentation_column: str = ""

    # A function to call to encode the data.
    # This should be a variant of the MedicalImage.encode method.
    on_fetch: EndpointProperty[OnFetchInterface]

    # We need to declare this here to enable the dynamic component
    # wrapper forwarding.
    # TODO: Add this to a generic CellComponent class.
    cell_info: Any = None
