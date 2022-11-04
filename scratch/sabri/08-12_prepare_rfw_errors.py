import json

import pandas as pd

path = "/Users/eyubogln/.meerkat/datasets/rfw/themis/rfw_image_demographics.json"
dct = json.load(open(path))
error_df = pd.DataFrame([{"filename": k, **v} for k, v in dct.items()])
error_df["image_id"] = error_df["filename"].str.rsplit(".", n=1).str[0]
error_df[["image_id", "v6_fnmr"]].to_csv(
    "/Users/eyubogln/.meerkat/datasets/rfw/themis/facecompare_v6_errors.csv",
    index=False,
)
# df = df.merge(error_df["image_id", "v6_fnmr", "v5_fnmr"], on="image_id")
# df["diff_fnmr"] = df["v6_fnmr"] - df["v5_fnmr"]
# return df
