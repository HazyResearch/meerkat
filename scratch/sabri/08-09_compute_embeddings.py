import meerkat as mk

dp = mk.get("rfw")
dp = mk.embed(dp, input="image", num_workers=0)
dp.write("rfw_embedded.csv")
