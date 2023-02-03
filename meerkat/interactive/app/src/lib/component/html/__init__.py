from typing import List, Optional

from pydantic import validator

from meerkat.interactive.app.src.lib.component.abstract import BaseComponent, Component, Slottable
from meerkat.interactive.endpoint import Endpoint
from meerkat.tools.utils import classproperty


class HtmlMixin:
    @classproperty
    def library(cls):
        return "html"

    @classproperty
    def namespace(cls):
        return "html"


class a(Slottable, HtmlMixin, Component):
    classes: Optional[str] = None
    style: Optional[str] = None

    href: str = ""
    target: str = ""
    rel: str = ""


class div(Slottable, HtmlMixin, Component):
    classes: Optional[str] = None
    style: Optional[str] = None

    def __init__(
        self, 
        slots: Optional[List[BaseComponent]] = None, 
        *, 
        classes: Optional[str] = None, 
        style: Optional[str] = None,
    ):
        super().__init__(slots=slots, classes=classes, style=style)

class flex(div):

    @validator("classes", pre=True, always=True)
    def make_flex(cls, v):
        return "flex flex-row " + v if v is not None else "flex flex-row"

class flexcol(div):
    
    @validator("classes", pre=True, always=True)
    def make_flexcol(cls, v):
        return "flex flex-col " + v if v is not None else "flex flex-col"

class grid(div):
    
    @validator("classes", pre=True, always=True)
    def make_grid(cls, v):
        return "grid " + v if v is not None else "grid"

class gridcols2(div):
    @validator("classes", pre=True, always=True)
    def make_gridcols2(cls, v):
        return "grid grid-cols-2 " + v if v is not None else "grid grid-cols-2"
    
class gridcols3(div):
    @validator("classes", pre=True, always=True)
    def make_gridcols3(cls, v):
        return "grid grid-cols-3 " + v if v is not None else "grid grid-cols-3"

class gridcols4(div):
    @validator("classes", pre=True, always=True)
    def make_gridcols4(cls, v):
        return "grid grid-cols-4 " + v if v is not None else "grid grid-cols-4"


class p(Slottable, HtmlMixin, Component):
    classes: Optional[str] = None
    style: Optional[str] = None


class span(Slottable, HtmlMixin, Component):
    classes: Optional[str] = None
    style: Optional[str] = None


class h1(Slottable, HtmlMixin, Component):
    classes: Optional[str] = "text-4xl"
    style: Optional[str] = None


class h2(Slottable, HtmlMixin, Component):
    classes: Optional[str] = "text-3xl"
    style: Optional[str] = None


class h3(Slottable, HtmlMixin, Component):
    classes: Optional[str] = "text-2xl"
    style: Optional[str] = None


class h4(Slottable, HtmlMixin, Component):
    classes: Optional[str] = "text-xl"
    style: Optional[str] = None


class h5(Slottable, HtmlMixin, Component):
    classes: Optional[str] = "text-lg"
    style: Optional[str] = None

class h6(Slottable, HtmlMixin, Component):
    classes: Optional[str] = "text-md"
    style: Optional[str] = None


# class radio(Slottable, HtmlMixin, Component):
#     classes: Optional[str] = None
#     style: Optional[str] = None

#     name: str = ""
#     value: str = ""
#     checked: bool = False
#     disabled: bool = False
#     color: str = "purple"

#     on_change: Optional[Endpoint] = None

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
