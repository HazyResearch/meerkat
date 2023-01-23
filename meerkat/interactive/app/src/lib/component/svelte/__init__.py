from meerkat.interactive.app.src.lib.component.abstract import AutoComponent, Slottable
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

# class fragment(Slottable, SvelteMixin, AutoComponent):

#     slot: str = None


# class options(Slottable, SvelteMixin, AutoComponent):

#     immutable: bool = False
#     accessors: bool = False
#     namespace: str = None
#     tag: str = None


# class head(Slottable, SvelteMixin, AutoComponent):

#     pass


# class component(Slottable, SvelteMixin, AutoComponent):

#     this: str = None
#     props: dict = None


# class self(Slottable, SvelteMixin, AutoComponent):

#     props: dict = None


# class each(Slottable, AutoComponent):

#     items: list = None
