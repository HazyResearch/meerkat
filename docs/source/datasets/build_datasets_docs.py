import os

import pandas as pd
from tabulate import tabulate

import meerkat as mk


def get_rst_class_ref(klass: type):
    return f":class:`dcbench.{klass.__name__}`"


def get_link(text: str, url: str):
    return f"`{text} <{url}>`_"


def get_datasets_table():
    dp = mk.datasets.catalog
    df = dp.to_pandas()
    df = df[["name", "homepage", "tags"]]
    style = df.style.set_table_styles(
        {"description": [{"selector": "", "props": "max-width: 50%;"}]}
    )
    df = df.set_index("name")

    html = style.to_html()

    return html


datasets_table = get_datasets_table()

open("source/datasets/datasets_table.html", "w").write(datasets_table)
