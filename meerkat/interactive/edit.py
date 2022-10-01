from dataclasses import dataclass

from .graph import Pivot, PivotConfig
from pydantic import BaseModel


class EditTargetConfig(BaseModel):
    target: PivotConfig
    target_id_column: str
    source_id_column: str


@dataclass
class EditTarget:
    target: Pivot
    target_id_column: str
    source_id_column: str

    @property
    def config(self):
        return EditTargetConfig(
            target=self.target.config,
            target_id_column=self.target_id_column,
            source_id_column=self.source_id_column,
        )
