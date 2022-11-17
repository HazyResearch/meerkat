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


import meerkat as mk

value = []


@mk.gui.endpoint
def reassign(value: mk.gui.Store):
    value.set(value + 1)
    print(value)

@mk.gui.reactive
def print_op(value: any):
    print("in print op")
    print(value)


with mk.gui.react():
    value = mk.gui.Store(0)
    button = mk.gui.Button("Increment", on_click=reassign(value))

    new_value = value + 10
    print_op(value)
    breakpoint()
    print_op(new_value)



mk.gui.start()
mk.gui.Interface(
    components=[button]
).launch()