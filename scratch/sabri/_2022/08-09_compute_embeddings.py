import meerkat as mk

df = mk.get("rfw")
df = mk.embed(df, input="image", num_workers=0)
df.write("rfw_embedded.csv")
