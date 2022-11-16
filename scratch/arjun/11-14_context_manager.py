"""Test the context manager with stores."""
# import meerkat as mk


# # Example 1
# s1 = mk.gui.Store(5)

# with mk.gui.react():
#     r = s1 + 2

# # the interface_op decorator should be applied to the addition method.
# # Thus:
# # 1. r should be a store because it is in mk.gui.react()
# # 2. r should be on the graph.
# assert isinstance(r, mk.gui.Store), type(r)
# # assert r.has_trigger_children()


import pandas as pd

import meerkat as mk

df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
df = mk.DataFrame.from_pandas(df)

@mk.gui.endpoint
def reassign(df: mk.DataFrame):
    df["c"] = df["b"]

print(df.inode)
# df = mk.gui.Reference(df)
with mk.gui.react():
    print("=============")
    columns_store = df.keys()

    print("Print inside ctx")
    print(list(df.inode.children.keys())[0].children)
    print(columns_store.inode)

columns = df.keys()
print(type(columns_store), columns_store)
print(columns, type(columns))

reassign(df).run()
print(df)
print(columns_store, type(columns_store))

# print(df.inode.children)
# print(columns_store.inode)

# import meerkat as mk

# a = mk.gui.Store(1)

# with mk.gui.react():
#     b = a + 1

# print(b, type(b))