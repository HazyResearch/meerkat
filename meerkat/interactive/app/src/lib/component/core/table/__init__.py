from typing import Optional

from pydantic import validator

from meerkat.dataframe import DataFrame
from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.endpoint import Endpoint


class Table(Component):
    df: DataFrame
    per_page: int = 100
    editable: bool = False
    id_column: Optional[str] = None

    on_edit: Endpoint = None

    # Create a Pydantic validator to ensure that the id_column is in the df
    # when editable is True
    @validator("id_column")
    def id_column_in_df(cls, v, values):
        if v is None and values["editable"]:
            raise ValueError("id_column must be specified when editable is True")
        return v
