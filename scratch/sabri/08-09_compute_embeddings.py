import meerkat as mk

dp = mk.get("imagenette")
dp = mk.embed(dp, input="img", num_workers=0)
dp.write("imagenette_embedded.csv")
