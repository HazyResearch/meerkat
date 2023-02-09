import rich

from meerkat.interactive.graph.reactivity import react

print = react()(rich.print)
