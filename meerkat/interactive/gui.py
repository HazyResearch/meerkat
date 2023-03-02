from typing import List

from IPython.display import IFrame

import meerkat as mk


class GUI:
    pass


class DataFrameGUI(GUI):
    def __init__(self, df: mk.DataFrame):
        self.df = df

    def table(
        self,
        **kwargs,
    ) -> IFrame:
        return mk.gui.Table(
            df=self.df,
            classes="h-[550px]",
            **kwargs,
        )

    def gallery(self, main_column: str = None, tag_columns: List[str] = None, **kwargs):
        return mk.gui.Gallery(
            df=self.df,
            main_column=main_column,
            tag_columns=tag_columns,
            **kwargs,
        )
