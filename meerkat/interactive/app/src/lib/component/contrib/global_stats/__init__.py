
from ...abstract import Component


class GlobalStats(Component):
    v1_name: str
    v2_name: str
    v1_mean: float
    v2_mean: float
    shift: float
    inconsistency: float
    metric: str = "Accuracy"
