# from typing import List, Union

# from meerkat.dataframe import DataFrame
# from meerkat.interactive.edit import EditTarget
# from meerkat.interactive.graph import Store, make_store

# from ..abstract import BaseComponent


# class StatsLabeler(BaseComponent):

#     name = "StatsLabeler"

#     def __init__(
#         self,
#         df: DataFrame,
#         label_target: EditTarget = None,
#         phase_target: EditTarget = None,
#         phase: Union[Store[str], str] = "train",
#         active_key: Union[Store[str], str] = None,
#         selected: Store[List[int]] = None,
#         primary_key: str = None,
#         precision_estimate: List[Store[float]] = None,
#         recall_estimate: List[Store[float]] = None,
#     ) -> None:
#         super().__init__()
#         self.df = df
#         self.label_target = label_target
#         self.phase_target = phase_target
#         self.phase = make_store(phase)
#         self.active_key = make_store(active_key)
#         self.selected = make_store(selected)
#         self.primary_key = primary_key
#         self.precision_estimate = make_store(precision_estimate)
#         self.recall_estimate = make_store(recall_estimate)

#     @property
#     def props(self):
#         return {
#             "df": self.df.config,  # FIXME
#             "label_target": self.label_target.config,
#             "phase_target": self.phase_target.config,
#             "phase": self.phase.config,
#             "active_key": self.active_key.config,
#             "selected": self.selected.config,
#             "primary_key": self.primary_key,
#             "precision_estimate": self.precision_estimate.config,
#             "recall_estimate": self.recall_estimate.config,
#         }
