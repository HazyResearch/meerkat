from typing import Callable, Dict, List, Union

from meerkat.dataframe import DataFrame
from meerkat.interactive.graph import Reference, make_ref, make_store
from meerkat.mixins.identifiable import IdentifiableMixin
from meerkat.ops.sliceby.sliceby import SliceBy

from ..abstract import Component


class Aggregation(IdentifiableMixin):

    identifiable_group: str = "aggregations"

    def __init__(self, func: Callable[["DataFrame"], Union[int, float, str]]):
        self.func = func
        super().__init__()

    def __call__(self, df: "DataFrame") -> Union[int, float, str]:
        return self.func(df)

    @property
    def config(self):
        return {
            "id": self.id,
        }


class SliceByCards(Component):

    name = "SliceByCards"

    def __init__(
        self,
        sliceby: Reference[SliceBy],
        main_column: str,
        tag_columns: List[str] = None,
        aggregations: Dict[str, Callable[["DataFrame"], Union[int, float, str]]] = None,
        df: Reference[
            "DataFrame"
        ] = None,  # required to support passing in an external ref
    ) -> None:
        super().__init__()
        self.sliceby = make_ref(sliceby)

        if df is None:
            df = self.sliceby.obj.data
        else:
            assert self.sliceby.obj.data is (
                df.obj if isinstance(df, Reference) else df
            )

        self.df = make_ref(df)

        if aggregations is None:
            aggregations = {}

        self.aggregations = {k: Aggregation(v) for k, v in aggregations.items()}

        self.main_column = make_store(main_column)

        if tag_columns is None:
            tag_columns = []

        self.tag_columns = make_store(tag_columns)

    @property
    def props(self):
        return {
            "sliceby": self.sliceby.config,
            "df": self.df.config,
            "main_column": self.main_column.config,
            "tag_columns": self.tag_columns.config,
            "aggregations": {k: v.config for k, v in self.aggregations.items()},
        }
