# from typing import List, Union

# import numpy as np

# from meerkat.dataframe import DataFrame
# from meerkat.interactive.app.src.lib.component.abstract import BaseComponent
# from meerkat.interactive.edit import EditTarget
# from meerkat.interactive.graph import Store, make_store


# class Editor(BaseComponent):

#     name = "Editor"

#     def __init__(
#         self,
#         df: DataFrame,
#         col: Union[Store, str],
#         target: EditTarget = None,
#         selected: Store[List[int]] = None,
#         primary_key: str = None,
#         title: str = None,
#     ) -> None:
#         super().__init__()
#         self.col = make_store(col)
#         self.text = make_store("")
#         self.primary_key = primary_key

#         self.df = df
#         if target is None:
#             df["_edit_id"] = np.arange(len(df))
#             target = EditTarget(self.df, "_edit_id", "_edit_id")
#         self.target = target
#         self.selected = selected
#         self.title = title if title is not None else ""

#     @property
#     def props(self):
#         return {
#             "df": self.df.config,  # FIXME
#             "target": self.target.config,
#             "col": self.col.config,
#             "text": self.text.config,
#             "selected": self.selected.config,
#             "primary_key": self.primary_key,
#             "title": self.title,
#         }
