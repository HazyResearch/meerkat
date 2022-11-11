from dataclasses import dataclass

from pydantic import BaseModel

from .graph import Reference, ReferenceConfig


class EditTargetConfig(BaseModel):
    target: ReferenceConfig
    target_id_column: str
    source_id_column: str


@dataclass
class EditTarget:
    target: Reference
    target_id_column: str
    source_id_column: str

    @property
    def config(self):
        return EditTargetConfig(
            target=self.target.config,
            target_id_column=self.target_id_column,
            source_id_column=self.source_id_column,
        )
