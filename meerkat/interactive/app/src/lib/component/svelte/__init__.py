from meerkat.interactive.app.src.lib.component.abstract import Component, Slottable
from meerkat.mixins.identifiable import classproperty


class SvelteMixin:
    @classproperty
    def library(cls):
        return "svelte"

    @classproperty
    def namespace(cls):
        return "svelte"

# class Lettable:

#     def __init__(self, let: dict = None, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.let = let

# class fragment(Slottable, SvelteMixin, Component):

#     slot: str = None


# class options(Slottable, SvelteMixin, Component):

#     immutable: bool = False
#     accessors: bool = False
#     namespace: str = None
#     tag: str = None


# class head(Slottable, SvelteMixin, Component):

#     pass


# class component(Slottable, SvelteMixin, Component):

#     this: str = None
#     props: dict = None


# class self(Slottable, SvelteMixin, Component):

#     props: dict = None


# class each(Slottable, Component):

#     items: list = None
