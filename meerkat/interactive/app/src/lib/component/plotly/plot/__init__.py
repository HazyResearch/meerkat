from meerkat.mixins.identifiable import classproperty
from ...abstract import Component


class Plot(Component):
    title: str

    @classproperty
    def namespace(cls):
        return "plotly"
