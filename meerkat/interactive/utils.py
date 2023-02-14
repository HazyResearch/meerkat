import rich

from meerkat.interactive.graph.reactivity import reactive

print = reactive(rich.print)
