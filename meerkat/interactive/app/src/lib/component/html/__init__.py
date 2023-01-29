from typing import Optional
from meerkat.interactive.app.src.lib.component.abstract import Component, Slottable
from meerkat.mixins.identifiable import classproperty


class HtmlMixin:
    @classproperty
    def library(cls):
        return "html"

    @classproperty
    def namespace(cls):
        return "html"


class p(Slottable, HtmlMixin, Component):
    classes: Optional[str] = None
    style: Optional[str] = None


class a(Slottable, HtmlMixin, Component):
    classes: Optional[str] = None
    style: Optional[str] = None

    href: str = ""
    target: str = ""
    rel: str = ""


class div(Slottable, HtmlMixin, Component):
    classes: Optional[str] = None
    style: Optional[str] = None


class span(Slottable, HtmlMixin, Component):
    classes: Optional[str] = None
    style: Optional[str] = None

# FIXME: remove closing tags in the Wrapper.svelte transipler
# class input(HtmlMixin, Component):
#     classes: Optional[str] = None
#     style: Optional[str] = None

#     type: str = "text"
#     value: str = ""
#     placeholder: str = ""
#     name: str = ""


# class img(Slottable, HtmlMixin, Component):
#     src: str = ""
#     alt: str = ""
#     width: str = ""
#     height: str = ""


class h1(Slottable, HtmlMixin, Component):
    classes: Optional[str] = None
    style: Optional[str] = None


class h2(Slottable, HtmlMixin, Component):
    pass


class h3(Slottable, HtmlMixin, Component):
    pass


class h4(Slottable, HtmlMixin, Component):
    pass


class h5(Slottable, HtmlMixin, Component):
    pass


class h6(Slottable, HtmlMixin, Component):
    pass

class svg(Slottable, HtmlMixin, Component):
    classes: Optional[str] = None
    style: Optional[str] = None

    fill: Optional[str] = None
    viewBox: Optional[str] = None
    stroke: Optional[str] = None
    stroke_width: Optional[str] = None
    stroke_linecap: Optional[str] = None
    stroke_linejoin: Optional[str] = None
    # Keeping this attribute in makes the svg component not render
    # xmlns: str = "http://www.w3.org/2000/svg"
    aria_hidden: Optional[str] = None

class path(Slottable, HtmlMixin, Component):
    classes: Optional[str] = None
    style: Optional[str] = None

    d: Optional[str] = None
    fill: Optional[str] = None
    clip_rule: Optional[str] = None
    fill_rule: Optional[str] = None
    stroke: Optional[str] = None
    stroke_linecap: Optional[str] = None
    stroke_linejoin: Optional[str] = None
    stroke_width: Optional[str] = None

class ul(Slottable, HtmlMixin, Component):
    classes: Optional[str] = None
    style: Optional[str] = None


class ol(Slottable, HtmlMixin, Component):
    pass


class li(Slottable, HtmlMixin, Component):
    pass


class table(Slottable, HtmlMixin, Component):
    pass


class thead(Slottable, HtmlMixin, Component):
    pass


class tbody(Slottable, HtmlMixin, Component):
    pass


class tr(Slottable, HtmlMixin, Component):
    pass


class th(Slottable, HtmlMixin, Component):
    pass


class td(Slottable, HtmlMixin, Component):
    pass


# class br(HtmlMixin, Component):
#     pass


# class hr(HtmlMixin, Component):
#     pass

class form(Slottable, HtmlMixin, Component):
    classes: Optional[str] = None
    style: Optional[str] = None

    action: Optional[str] = None
    method: str = "get"
    enctype: str = "application/x-www-form-urlencoded"
    target: Optional[str] = None

class button(Slottable, HtmlMixin, Component):
    
    classes: Optional[str] = None
    style: Optional[str] = None

    type: str = "button"
    value: Optional[str] = None
    name: Optional[str] = None
    disabled: bool = False

    formaction: Optional[str] = None


# class input(Slottable, HtmlMixin, Component):
#     pass


class textarea(Slottable, HtmlMixin, Component):
    pass


class select(Slottable, HtmlMixin, Component):
    pass


# class option(Slottable, HtmlMixin, Component):
#     pass


# class label(Slottable, HtmlMixin, Component):
#     pass


# class form(Slottable, HtmlMixin, Component):
#     pass


# class iframe(Slottable, HtmlMixin, Component):
#     pass


# class script(Slottable, HtmlMixin, Component):
#     pass


# class style(Slottable, HtmlMixin, Component):
#     pass


# class link(Slottable, HtmlMixin, Component):
#     pass


# class meta(Slottable, HtmlMixin, Component):
#     pass


# class header(Slottable, HtmlMixin, Component):
#     pass


# class footer(Slottable, HtmlMixin, Component):
#     pass


# class nav(Slottable, HtmlMixin, Component):
#     pass


# class main(Slottable, HtmlMixin, Component):
#     pass


# class section(Slottable, HtmlMixin, Component):
#     pass


# class article(Slottable, HtmlMixin, Component):
#     pass


# class aside(Slottable, HtmlMixin, Component):
#     pass


# class details(Slottable, HtmlMixin, Component):
#     pass


# class summary(Slottable, HtmlMixin, Component):
#     pass


# class dialog(Slottable, HtmlMixin, Component):
#     pass


# class menu(Slottable, HtmlMixin, Component):
#     pass


# class menuitem(Slottable, HtmlMixin, Component):
#     pass
