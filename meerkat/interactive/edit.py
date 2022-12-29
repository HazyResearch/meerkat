from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from meerkat.dataframe import DataFrame


class EditTargetConfig(BaseModel):
    # FIXME: this used to be ReferenceConfig
    # (ref_id: str, is_store = True, type = "DataFrame")
    target: Any
    target_id_column: str
    source_id_column: str


@dataclass
class EditTarget:
    target: DataFrame  # FIXME: is this right?
    target_id_column: str
    source_id_column: str

    @property
    def config(self):
        return EditTargetConfig(
            target=self.target.config,  # FIXME
            target_id_column=self.target_id_column,
            source_id_column=self.source_id_column,
        )
