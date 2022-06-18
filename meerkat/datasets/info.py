from dataclasses import dataclass
from datasets import DatasetInfo


@dataclass
class DatasetInfo:

    name: str
    description: str = None
    citation: str = None
    homepage: str = None
    license: str = None
    tags: str = None
