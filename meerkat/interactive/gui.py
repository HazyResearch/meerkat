from typing import Callable, Dict, List, Union

from IPython.display import IFrame

import meerkat as mk
from meerkat.interactive.app.src.lib.component.core.slicebycards import SliceByCards
from meerkat.mixins.identifiable import IdentifiableMixin
from meerkat.ops.sliceby.sliceby import SliceBy

from . import Page


class GUI:
    pass


class DataFrameGUI(GUI):
    def __init__(self, df: mk.DataFrame):
        self.df = df

    def table(
        self,
        **kwargs,
    ) -> IFrame:
        return Page(
            component=mk.gui.Table(
                df=self.df,
                **kwargs,
            ),
            id="table",
        ).launch()

    def gallery(self, main_column: str = None, tag_columns: List[str] = None, **kwargs):
        if tag_columns is None:
            tag_columns = []
        if main_column is None:
            main_column = self.df.columns[0]

        return Page(
            component=mk.gui.Gallery(
                df=self.df,
                main_column=main_column,
                tag_columns=tag_columns,
                **kwargs,
            ),
            id="gallery",
        ).launch()


class SliceByGUI(GUI):
    def __init__(self, sb: SliceBy):
        self.sb = sb

    def cards(
        self,
        main_column: str,
        tag_columns: List[str] = None,
        aggregations: Dict[
            str, Callable[[mk.DataFrame], Union[int, float, str]]
        ] = None,
    ) -> IFrame:
        """_summary_

        Args:
            main_column (str): This column will be shown.
            tag_columns (List[str], optional): _description_. Defaults to None.
            aggregations (Dict[
                str, Callable[[mk.DataFrame], Union[int, float, str]]
            ], optional): A dictionary mapping from aggregation names to functions
                that aggregate a DataFrame. Defaults to None.

        Returns:
            IFrame: _description_
        """
        component = SliceByCards(
            sliceby=self.sb,
            main_column=main_column,
            tag_columns=tag_columns,
            aggregations=aggregations,
        )
        return mk.gui.Page(component=component, id="sliceby").launch()


class Aggregation(IdentifiableMixin):
    _self_identifiable_group: str = "aggregations"

    def __init__(self, func: Callable[[mk.DataFrame], Union[int, float, str]]):
        self.func = func
        super().__init__()

    def __call__(self, df: mk.DataFrame) -> Union[int, float, str]:
        return self.func(df)
