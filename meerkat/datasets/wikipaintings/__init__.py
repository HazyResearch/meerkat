import datetime
import os

import PIL
import requests

import meerkat as mk
from meerkat.columns.deferred.image import load_image

from ..abstract import DatasetBuilder
from ..info import DatasetInfo
from ..registry import datasets

REPO = "https://github.com/NationalGalleryOfArt/opendata.git"


def _write_empty_image(dst):
    img = PIL.Image.new("RGB", (32, 32), color="black")
    img.save(dst, format="JPEG")


@datasets.register()
class wikipaintings(DatasetBuilder):
    VERSIONS = ["main"]

    info = DatasetInfo(
        name="wikipaintings",
        full_name="Paintings from WikiData",
        # flake8: noqa
        description="",
        homepage="https://www.wikidata.org/wiki/Wikidata:Main_Page",
        tags=["art"],
        citation=None,
    )

    def build(self):
        df = mk.read(os.path.join(self.dataset_dir, "data.mk"))
        df = df[~df["qid"].duplicated()]
        df = df[~df["title"].duplicated()]

        return df

    def download(self):
        url = "https://query.wikidata.org/sparql"
        query = """
        SELECT ?painting ?paintingLabel ?image ?date
        WHERE {
        ?painting wdt:P31 wd:Q3305213.
        ?painting wdt:P170 ?artist.
        ?painting wdt:P18 ?image. 
        ?painting wdt:P571 ?date.
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        """
        r = requests.get(url, params={"format": "json", "query": query})
        data = r.json()

        def extract_year(date: str):
            try:
                return datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ").year
            except ValueError:
                return -1

        df = mk.DataFrame(
            [
                {
                    "qid": row["painting"]["value"].split("/")[-1],
                    "title": row["paintingLabel"]["value"],
                    "image_url": row["image"]["value"],
                    "year": extract_year(row["date"]["value"]),
                    "artist": row["artistLabel"]["value"],
                }
                for row in data["results"]["bindings"]
            ]
        )
        df.write(os.path.join(self.dataset_dir, "data.mk"))
