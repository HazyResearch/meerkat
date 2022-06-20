from typing import List

import meerkat as mk


def get_rst_class_ref(klass: type):
    return f":class:`dcbench.{klass.__name__}`"


def get_link(text: str, url: str):
    return f"`{text} <{url}>`_"


def create_tags_html(tags: List[str]):
    tags = "".join([f"<div class='tag'>{tag.replace('_', ' ')}</div>" for tag in tags])
    html = f"""
        <div class='tags'>
            {tags}
        </div>
    """
    return html


def create_versions_html(versions: List[str]):
    versions = "".join(
        [f"<div class='tag'>{tag.replace('_', ' ')}</div>" for tag in versions]
    )
    html = f"""
        <div class='versions'>
            {versions}
        </div>
    """
    return html


def get_datasets_table():
    dp = mk.datasets.catalog
    df = dp.to_pandas()
    df["versions"] = df["name"].apply(
        lambda x: create_versions_html(mk.datasets.versions(x))
    )
    df["homepage"] = df["homepage"].apply(lambda x: f'<a href="{x}">link</a>')
    df["tags"] = df["tags"].apply(create_tags_html)
    df = df.set_index("name")
    df.index.name = None

    style = df[["tags", "versions", "homepage"]].style.set_table_styles(
        {"description": [{"selector": "", "props": "max-width: 50%;"}]}
    )

    html = style.to_html(escape=False)

    html += """
        <style>
            .tag {
                font-family: monospace;
                font-size: 0.8em;
                border: 10px;
                border-radius: 5px;
                border-color: black;
                padding-left: 7px;
                padding-right: 7px;
                padding-top: 0px;
                padding-bottom: 0px;
                margin: 1px;
            }
            .tags .tag {
                background-color: lightgrey;
            }
            .versions .tag {
                background-color: lightblue;
            }
            .tags {
                display: inline-flex;
                flex-wrap: wrap;
            }
            .versions {
                display: inline-flex;
                flex-wrap: wrap;
            }

        </style>
    """
    return html


datasets_table = get_datasets_table()

open("source/datasets/datasets_table.html", "w").write(datasets_table)
