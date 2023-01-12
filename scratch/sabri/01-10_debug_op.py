import meerkat as mk 
df = mk.get("imagenette")
df[:10].map(lambda path: {"x": [1, 1], "y": [2, 2]}, is_batched_fn=True, batch_size=2)