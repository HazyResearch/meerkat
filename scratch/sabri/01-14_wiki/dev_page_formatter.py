import requests
import pywikibot as pwb
import os
import meerkat as mk
from meerkat.interactive.formatter.base import WebsiteFormatter


url = "https://query.wikidata.org/sparql"
query = """
SELECT ?country ?countryLabel ?article
WHERE {
  ?country wdt:P31 wd:Q6256.
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT 10
"""
r = requests.get(url, params={"format": "json", "query": query})
data = r.json()


df = mk.DataFrame(
    [
        {"uri": row["country"]["value"], "country": row["countryLabel"]["value"]}
        for row in data["results"]["bindings"]
    ]
)
df["uri"].formatter = WebsiteFormatter()
df["qid"] = df["uri"].map(lambda x: os.path.basename(x))

def get_wikipedia_title(qid: str):
    site = pwb.Site("wikipedia:en")
    repo = site.data_repository()
    item = pwb.ItemPage(repo, qid)
    title = item.sitelinks["enwiki"].title
    return title

df["title"] = df.map(get_wikipedia_title, pbar=True)
df["wikipedia_url"] = df.map(
    lambda title: f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
)
df["wikipedia_url"].formatter = WebsiteFormatter()


mk.gui.start()


df.gui.gallery(main_column="wikipedia_url", tag_columns=["country"])
