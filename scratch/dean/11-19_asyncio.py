import aiohttp
from aiohttp import ClientSession
import argparse
import asyncio
import requests
import time


URLS = [
    "https://regex101.com/",
    "https://www.nytimes.com/guides/",
    "https://www.mediamatters.org/",
    "https://1.1.1.1/",
    "https://www.politico.com/tipsheets/morning-money",
    "https://www.bloomberg.com/markets/economics",
    "https://www.ietf.org/rfc/rfc2616.txt"
]


def get_html_seq(url: str) -> str:
    res = requests.get(url=url)
    res.raise_for_status()
    print(f"{url} gave status {res.status_code}")
    html = res.text
    return html


def main_seq(urls: set[str]) -> None:
    for url in urls:
        get_html_seq(url=url)


async def get_html_async(url: str, session: ClientSession) -> str:
    res = await session.request(method="GET", url=url)
    res.raise_for_status()
    print(f"{url} gave status {res.status}")
    html = await res.text()
    return html


async def main_async(urls: set[str]) -> None:
    async with ClientSession() as session:
        tasks = [get_html_async(url=url, session=session) for url in urls]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make get requests to urls")
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--sequential', dest='parallel', action='store_false')
    parser.set_defaults(parallel=True)
    args = parser.parse_args()

    start = time.perf_counter()
    if args.parallel:
        asyncio.run(main_async(urls=URLS))
    else:
        main_seq(urls=URLS)
    end = time.perf_counter()

    print(f"Program took {end - start:0.2f} seconds ", end="")
    print(f"using {'parallel' if args.parallel else 'sequential'}.")
