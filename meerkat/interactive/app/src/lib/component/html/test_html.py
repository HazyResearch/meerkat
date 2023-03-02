import functools
import os

import pytest

from meerkat.interactive import Page
from meerkat.interactive.app.src.lib.component.html import (
    div,
    flex,
    flexcol,
    grid,
    gridcols2,
    gridcols3,
    gridcols4,
    h1,
    h2,
    h3,
    h4,
    h5,
    h6,
)


def hello_div():
    return div(slots=["Hello world!"])


def hello_div_k(k=20):
    return [hello_div()] * k


def test_div():
    return functools.partial(div, slots=hello_div_k())


def test_flex():
    return functools.partial(flex, slots=hello_div_k())


def test_flexcol():
    return functools.partial(flexcol, slots=hello_div_k())


def test_grid():
    return functools.partial(grid, slots=hello_div_k())


def test_gridcols2():
    return functools.partial(gridcols2, slots=hello_div_k())


def test_gridcols3():
    return functools.partial(gridcols3, slots=hello_div_k())


def test_gridcols4():
    return functools.partial(gridcols4, slots=hello_div_k())


@pytest.mark.skipif(os.environ.get("TEST_REMOTE", False), reason="Skip html tests")
def test_html_components():
    component = div(
        slots=[
            "Div with flex-col",
            test_div()(classes="flex flex-col bg-slate-200 text-red-800"),
            "Div with flex-row",
            test_div()(classes="flex flex-row"),
            "Div with grid-cols-2",
            test_div()(classes="grid grid-cols-2"),
            "Flex",
            test_flex()(),
            "Flexcol",
            test_flexcol()(),
            "Grid",
            test_grid()(),
            "Gridcols2",
            test_gridcols2()(),
            "Gridcols3",
            test_gridcols3()(),
            "Gridcols4",
            test_gridcols4()(),
            "H1",
            h1(slots=["H1"]),
            "H2",
            h2(slots=["H2"]),
            "H3",
            h3(slots=["H3"]),
            "H4",
            h4(slots=["H4"]),
            "H5",
            h5(slots=["H5"]),
            "H6",
            h6(slots=["H6"]),
        ]
    )
    page = Page(component=component, id="html")
    page.launch()
