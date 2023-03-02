from typing import Callable, Dict, List, Union

from meerkat.dataframe import DataFrame
from meerkat.interactive.app.src.lib.component.abstract import BaseComponent
from meerkat.interactive.graph import make_store
from meerkat.mixins.identifiable import IdentifiableMixin
from meerkat.ops.sliceby.sliceby import SliceBy


class Aggregation(IdentifiableMixin):
    _self_identifiable_group: str = "aggregations"

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


class SliceByCards(BaseComponent):  # FIXME: update this component
    def __init__(
        self,
        sliceby: SliceBy,
        main_column: str,
        tag_columns: List[str] = None,
        aggregations: Dict[str, Callable[["DataFrame"], Union[int, float, str]]] = None,
        df: DataFrame = None,
    ) -> None:
        super().__init__()
        self.sliceby = sliceby

        if df is None:
            df = self.sliceby.obj.data
        else:
            assert self.sliceby.obj.data is df

        self.df = df

        if aggregations is None:
            aggregations = {}

        self.aggregations = {k: Aggregation(v) for k, v in aggregations.items()}

        self.main_column = make_store(main_column)

        if tag_columns is None:
            tag_columns = []

        self.tag_columns = make_store(tag_columns)
