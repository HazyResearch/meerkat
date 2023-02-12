import rich

from meerkat.interactive.graph.reactivity import _reactive

print = _reactive(rich.print)
