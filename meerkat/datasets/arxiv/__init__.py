import os
import pyarrow as pa
import pyarrow.compute as pc

import meerkat as mk

from ..abstract import DatasetBuilder
from ..info import DatasetInfo
from ..registry import datasets
from ..utils import extract


@datasets.register()
class arxiv(DatasetBuilder):

    VERSIONS = ["full"]

    info = DatasetInfo(
        name="arxiv",
        full_name="arXiv Dataset",
        description=(
            "This dataset contains metadata for all 1.7 million+ arXiv articles "
            "from the period of 1991 to 2021 across various research fields."
        ),
        homepage="https://www.kaggle.com/Cornell-University/arxiv",
        tags=["arxiv", "metadata", "research", "papers"],
    )

    @property
    def data_dir(self):
        return os.path.join(self.dataset_dir, "arxiv-metadata-oai-snapshot")

    def build(self):
        df = mk.from_json(
            os.path.join(self.dataset_dir, "arxiv-metadata-oai-snapshot.json"),
            lines=True,
            backend="arrow",
        )
        df.set_primary_key("id", inplace=True)

        # TODO: This is a hack to remove the rows that are not in main arxiv.
        df = df[~df["id"].str.contains("/")]

        df["latest"] = "v" + mk.ArrowScalarColumn(
            pc.cast(pc.list_value_length(df["versions"]._data), pa.string())
        )
        df["pdf_uri"] = (
            "gs://arxiv-dataset/arxiv/arxiv/pdf/"
            + df["id"].str.split(".")["0"]
            + "/"
            + df["id"]
            + df["latest"]
            + ".pdf"
        )
        df["pdf"] = mk.files(
            df["pdf_uri"],
            cache_dir=os.path.join(self.dataset_dir, "pdfs"),
            fallback_downloader=lambda dst: open(dst, "wb").write(b""),
        )

        try:

            df["text"] = (
                df["pdf"]
                .defer(extract_text)
                .format(
                    mk.format.TextFormatterGroup().update(
                        dict(tiny=mk.format.IconFormatter(name="FileEarmarkFont"))
                    ).defer()
                )
            )

        except ImportError:
            pass

        return df

    def download(self):

        self.download_kaggle_dataset("Cornell-University/arxiv", self.dataset_dir)
        extract(
            os.path.join(self.dataset_dir, "arxiv.zip"),
            self.dataset_dir,
        )

    @staticmethod
    def download_kaggle_dataset(dataset, dest_dir):
        import kaggle.api

        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset, path=dest_dir, unzip=False)


def extract_text(bytes: bytes) -> str:
    import pdftotext
    import io

    pdf = pdftotext.PDF(io.BytesIO(bytes))
    return "\n\n".join(pdf)
