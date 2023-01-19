import ray
import pandas as pd
import numpy as np
import time

import meerkat as mk

ray.init()

df = mk.get("imagenette", version="160px")

args = df["img"].data.args
ds = ray.data.from_pandas(
    pd.DataFrame({str(idx): arg.to_pandas() for idx, arg in enumerate(args)})
)
ds = ds.repartition(100)


def load(x):
    # print(f"loading path: {x}")
    return np.array(df["img"].data.fn(x["0"]))


def operation(x):
    # print("operation")
    x = x.mean()
    return x


pipeline = ds.window(blocks_per_window=10).map(load).map(operation)

# time the code pipeline.take_all() code below
start = time.time()
pipeline.take_all()
end = time.time()
print(f"Time taken: {end - start}")


paths = list("/Users/deanstratakos/.meerkat/datasets/imagenette/160px/imagenette2-160/" + args[0])
ds = ray.data.read_images(paths)


def operation(x):
    x = x["image"].mean()
    return x


pipeline = ds.window(blocks_per_window=10).map(operation)

# time the code pipeline.take_all() code below
start = time.time()
pipeline.take_all()
end = time.time()
print(f"Time taken: {end - start}")
