from meerkat.interactive.app.src.lib.component.abstract import AutoComponent, Slottable
from meerkat.mixins.identifiable import classproperty


class HtmlMixin:
    @classproperty
    def library(cls):
        return "html"

    @classproperty
    def namespace(cls):
        return "html"


class p(Slottable, HtmlMixin, AutoComponent):
    pass


class a(Slottable, HtmlMixin, AutoComponent):
    href: str = ""


class div(Slottable, HtmlMixin, AutoComponent):
    pass


class span(Slottable, HtmlMixin, AutoComponent):
    pass


# class img(Slottable, HtmlMixin, AutoComponent):
#     src: str = ""
#     alt: str = ""
#     width: str = ""
#     height: str = ""


class h1(Slottable, HtmlMixin, AutoComponent):
    classes: str = ""
    style: str = ""


class h2(Slottable, HtmlMixin, AutoComponent):
    pass


class h3(Slottable, HtmlMixin, AutoComponent):
    pass


class h4(Slottable, HtmlMixin, AutoComponent):
    pass


class h5(Slottable, HtmlMixin, AutoComponent):
    pass


class h6(Slottable, HtmlMixin, AutoComponent):
    pass


class ul(Slottable, HtmlMixin, AutoComponent):
    pass


class ol(Slottable, HtmlMixin, AutoComponent):
    pass


class li(Slottable, HtmlMixin, AutoComponent):
    pass


class table(Slottable, HtmlMixin, AutoComponent):
    pass


class thead(Slottable, HtmlMixin, AutoComponent):
    pass


class tbody(Slottable, HtmlMixin, AutoComponent):
    pass


class tr(Slottable, HtmlMixin, AutoComponent):
    pass


class th(Slottable, HtmlMixin, AutoComponent):
    pass


class td(Slottable, HtmlMixin, AutoComponent):
    pass


# class br(HtmlMixin, AutoComponent):
#     pass


# class hr(HtmlMixin, AutoComponent):
#     pass


class button(Slottable, HtmlMixin, AutoComponent):
    pass


# class input(Slottable, HtmlMixin, AutoComponent):
#     pass


class textarea(Slottable, HtmlMixin, AutoComponent):
    pass


class select(Slottable, HtmlMixin, AutoComponent):
    pass


# class option(Slottable, HtmlMixin, AutoComponent):
#     pass


# class label(Slottable, HtmlMixin, AutoComponent):
#     pass


# class form(Slottable, HtmlMixin, AutoComponent):
#     pass


# class iframe(Slottable, HtmlMixin, AutoComponent):
#     pass


# class script(Slottable, HtmlMixin, AutoComponent):
#     pass


# class style(Slottable, HtmlMixin, AutoComponent):
#     pass


# class link(Slottable, HtmlMixin, AutoComponent):
#     pass


# class meta(Slottable, HtmlMixin, AutoComponent):
#     pass


# class header(Slottable, HtmlMixin, AutoComponent):
#     pass


# class footer(Slottable, HtmlMixin, AutoComponent):
#     pass


# class nav(Slottable, HtmlMixin, AutoComponent):
#     pass


# class main(Slottable, HtmlMixin, AutoComponent):
#     pass


# class section(Slottable, HtmlMixin, AutoComponent):
#     pass


# class article(Slottable, HtmlMixin, AutoComponent):
#     pass


# class aside(Slottable, HtmlMixin, AutoComponent):
#     pass


# class details(Slottable, HtmlMixin, AutoComponent):
#     pass


# class summary(Slottable, HtmlMixin, AutoComponent):
#     pass


# class dialog(Slottable, HtmlMixin, AutoComponent):
#     pass


# class menu(Slottable, HtmlMixin, AutoComponent):
#     pass


# class menuitem(Slottable, HtmlMixin, AutoComponent):
#     pass
