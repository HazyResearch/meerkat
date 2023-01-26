from ..abstract import AutoComponent


class Text(AutoComponent):

    data: str
    dtype: str = None
    precision: int = 3
    percentage: bool = False
