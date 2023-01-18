from typing import List
import datetime 
import time 
import requests

SEARCH_TEMPLATE = "https://www.wikipedia.org/w/api.php?action=query&list=search&srsearch={term}&format=json"


def search(term: str):
    """Search for wikipedia page by their titles.
    Uses API described: https://www.mediawiki.org/wiki/API:Prefixsearch
    """
    SEARCH_TEMPLATE.format(term=term)
    response = requests.get(SEARCH_TEMPLATE.format(term=term))
    return response.json()["query"]["search"]


def link(name: str, tags: List[str]):
    results = search(term=name)
    qid = "none"
    if len(results) == 0:
        return qid
    for result in results:
        for tag in tags:
            if tag in result["snippet"]:
                qid = result["title"]
                return result
    return qid


def get_page(page_id: str):
    """Get wikipedia page by page id."""
    url = "https://www.wikidata.org/w/api.php?action=parse&format=json&page={page_id}"
    response = requests.get(url.format(page_id=page_id))
    return response.json()  ##["parse"]["text"]["*"]


API_URL = "http://en.wikipedia.org/w/api.php"
RATE_LIMIT = False
RATE_LIMIT_MIN_WAIT = None
RATE_LIMIT_LAST_CALL = None
USER_AGENT = "wikipedia (https://github.com/goldsmith/Wikipedia/)"


def _wiki_request(params):
    """
    Make a request to the Wikipedia API using the given search parameters.
    Returns a parsed dict of the JSON response.
    """
    global RATE_LIMIT_LAST_CALL
    global USER_AGENT

    params["format"] = "json"
    if "action" not in params:
        params["action"] = "query"

    headers = {"User-Agent": USER_AGENT}

    if (
        RATE_LIMIT
        and RATE_LIMIT_LAST_CALL
        and RATE_LIMIT_LAST_CALL + RATE_LIMIT_MIN_WAIT > datetime.now()
    ):

        # it hasn't been long enough since the last API call
        # so wait until we're in the clear to make the request

        wait_time = (RATE_LIMIT_LAST_CALL + RATE_LIMIT_MIN_WAIT) - datetime.now()
        time.sleep(int(wait_time.total_seconds()))

    r = requests.get(API_URL, params=params, headers=headers)
    return r 

    if RATE_LIMIT:
        RATE_LIMIT_LAST_CALL = datetime.now()

    return r.json()
