from meerkat.interactive.app.src.lib.component.abstract import Component


class Vega(Component):
    data: dict
    spec: dict
    options: dict = {}
