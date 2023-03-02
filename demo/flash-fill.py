"""Flash-fill.

Run this script with:
    OPENAI_API_KEY`mk run flash-fill.py`.

Requirements:
    pip install manifest-ml
"""
import os

import meerkat as mk

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ARXIV_DATASET_JSON = os.getenv("ARXIV_DATASET_JSON")
PDF_CACHE = os.getenv("PDF_CACHE", "~/.meerkat/cache/arxiv-dataset/pdfs")

if not ARXIV_DATASET_JSON:
    raise ValueError(
        "Please download the ArXiv Dataset from:\n"
        "https://www.kaggle.com/datasets/Cornell-University/arxiv/versions/5?resource=download\n"  # noqa: E501
        "and set the environment variable `ARXIV_DATASET_JSON` "
        "to the path of the snapshot file."
    )
ARXIV_DATASET_JSON = os.path.abspath(os.path.expanduser(ARXIV_DATASET_JSON))
PDF_CACHE = os.path.abspath(os.path.expanduser(PDF_CACHE))


filepath = ARXIV_DATASET_JSON
df = mk.from_json(filepath=filepath, lines=True, backend="arrow")
df = df[df["categories"].str.contains("stat.ML")]
df = mk.from_pandas(df.to_pandas(), primary_key="id")
df["url"] = "https://arxiv.org/pdf/" + df["id"]
df["pdf"] = mk.files(df["url"], cache_dir=PDF_CACHE, type="pdf")

df = df[["id", "authors", "title", "abstract", "pdf"]]
df["answer"] = ""

page = mk.gui.Page(
    component=mk.gui.contrib.FlashFill(df=df, target_column="answer"),
    id="flash-fill",
    progress=False,
)

page.launch()
