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

    on_edit: Optional[Endpoint] = None

    def __init__(
        self,
        df: DataFrame,
        *,
        per_page: int = 100,
        editable: bool = False,
        id_column: Optional[str] = None,
        on_edit: Optional[Endpoint] = None,
    ):
        super().__init__(
            df=df,
            per_page=per_page,
            editable=editable,
            id_column=id_column,
            on_edit=on_edit,
        )

    # Create a Pydantic validator to ensure that the id_column is in the df
    # when editable is True
    @validator("id_column")
    def id_column_in_df(cls, v, values):
        if v is None and values["editable"]:
            raise ValueError("id_column must be specified when editable is True")
        return v
