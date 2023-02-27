"""Flash-fill."""
import re
from functools import partial

from manifest import Manifest

import meerkat as mk

# manifest = Manifest(
#     client_name="huggingface",
#     client_connection="http://127.0.0.1:7861",
# )
manifest = Manifest(
    client_name="openai",
    client_connection=open("/Users/sabrieyuboglu/.meerkat/keys/.openai").read(),
    engine="code-davinci-002",
    temperature=0,
    max_tokens=1
)

def complete_prompt(row, example_template: mk.Store[str]):
    assert isinstance(row, dict)
    output = example_template.format(**row)
    return output


filepath = "/Users/sabrieyuboglu/Downloads/arxiv-metadata-oai-snapshot.json"
df = mk.from_json(filepath=filepath, lines=True, backend="arrow")
df = df[df["categories"].str.contains("stat.ML")]
df = mk.from_pandas(df.to_pandas(), primary_key="id")
df["url"] = "https://arxiv.org/pdf/" + df["id"]
df["pdf"] = mk.files(
    df["url"], cache_dir="/Users/sabrieyuboglu/Downloads/pdf-cache", type="pdf"
)

df = df[
    ["id", "authors", "title", "abstract", "pdf"]
]
df["answer"] = ""

page = mk.gui.Page(
    component=mk.gui.contrib.FlashFill(df=df, target_column="answer"),
    id="flash-fill",
    progress=False,
)

page.launch()


""" 
Does this paper report empirical results on real-world datasets?

Title: {title}
Abstract: {abstract}
Answer: {theoretical}

Yes
Yes
Yes
No
Yes
No
"""

