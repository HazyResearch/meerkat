"""Test the context manager with stores."""
import meerkat as mk


# Example 1
s1 = mk.gui.Store(5)

with mk.gui.react():
    r = s1 + 2

# the interface_op decorator should be applied to the addition method.
# Thus:
# 1. r should be a store because it is in mk.gui.react()
# 2. r should be on the graph.
assert isinstance(r, mk.gui.Store), type(r)
# assert r.has_trigger_children()