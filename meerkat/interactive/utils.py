import json
from typing import Any
import rich

from meerkat.interactive.graph.reactivity import react
from meerkat.columns.abstract import Column


print = react()(rich.print)


class MeerkatJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any):
        if isinstance(obj, Column):
            return [cell for cell in obj]
