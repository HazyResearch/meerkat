from dataclasses import dataclass, field
from typing import List

from meerkat.dataframe import DataFrame
from meerkat.interactive.graph import Store, make_store

from ..abstract import Component


@dataclass
class EditTarget:
    df: DataFrame
    pivot_id_column: str
    id_column: str

    @property
    def config(self):
        return {
            "pivot": self.pivot.config,
            "pivot_id_column": self.pivot_id_column,
            "id_column": self.id_column,
        }


@dataclass
class Gallery(Component):

    df: DataFrame
    main_column: str
    tag_columns: List[str] = field(default_factory=list)
    selected: List[int] = field(default_factory=list)



# class Gallery(Component):

#     name = "Gallery"

#     def __init__(
#         self,
#         df: DataFrame,
#         main_column: str,
#         tag_columns: List[str],
#         edit_target: EditTarget = None,
#         selected: Store[List[int]] = None,
#         primary_key: str = None,
#     ) -> None:
#         super().__init__()
#         self.df = df
#         self.main_column = make_store(main_column)
#         self.tag_columns = make_store(tag_columns)
#         self.primary_key = primary_key

#         if edit_target is None:
#             # TODO: primary key - make this based on primary keys once that is
#             # implemented
#             edit_target = EditTarget(self.df, self.primary_key, self.primary_key)
#         self.edit_target = edit_target

#         if primary_key is None:
#             primary_key = df._._primary_key
#         self.primary_key = primary_key

#         if selected is None:
#             selected = []
#         self.selected = make_store(selected)

#     @property
#     def props(self):
#         props = {
#             "df": self.df.config,  # FIXME
#             "main_column": self.main_column.config,
#             "tag_columns": self.tag_columns.config,
#             "edit_target": self.edit_target.config,
#             "selected": self.selected.config,
#             "primary_key": self.primary_key,
#         }
#         return props
