import os

import meerkat as mk


def display_df(df: mk.DataFrame, name: str):
    # need to get absolute paths so this works on readthedocs
    base_dir = os.path.join(os.path.dirname(os.path.dirname(mk.__file__)), "docs")
    body_html = df._repr_html_()
    css = open(os.path.join(base_dir, "source/html/display/dataframe.css"), "r").read()
    body_html = body_html.replace("\n", f"\n <style> {css} </style>", 1)
    open(os.path.join(base_dir, f"source/html/display/{name}.html"), "w").write(
        body_html
    )
    return df
