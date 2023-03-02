from dataclasses import dataclass


@dataclass
class DatasetInfo:
    name: str
    full_name: str = None
    description: str = None
    citation: str = None
    homepage: str = None
    license: str = None
    tags: str = None
