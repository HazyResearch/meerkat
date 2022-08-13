import json

import pandas as pd

path = "/Users/eyubogln/.meerkat/datasets/rfw/themis/rfw_image_demographics.json"
dct = json.load(open(path))
error_dp = pd.DataFrame([{"filename": k, **v} for k, v in dct.items()])
error_dp["image_id"] = error_dp["filename"].str.rsplit(".", n=1).str[0]
error_dp[["image_id", "v6_fnmr"]].to_csv(
    "/Users/eyubogln/.meerkat/datasets/rfw/themis/facecompare_v6_errors.csv",
    index=False,
)
# dp = dp.merge(error_dp["image_id", "v6_fnmr", "v5_fnmr"], on="image_id")
# dp["diff_fnmr"] = dp["v6_fnmr"] - dp["v5_fnmr"]
# return dp
