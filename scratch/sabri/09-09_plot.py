from typing import List

import numpy as np

import meerkat as mk
from meerkat.interactive.graph import Store

dp = mk.get("celeba")


def plot(dp: mk.DataPanel) -> mk.gui.Interface:
    pivot = mk.gui.Pivot(dp)
    selection = mk.gui.Store([])

    @mk.gui.interface_op
    def sliceby(dp: mk.DataPanel):
        sb = dp.sliceby(
            [
                "male",
                "brown_hair",
                "high_cheekbones",
                "gray_hair",
                "bangs",
                "eyeglasses",
                "pale_skin",
                "wearing_hat",
                "wearing_earrings",
                "wearing_necktie",
                "young",
                "straight_hair",
                "bald",
                "sideburns",
                "mustache",
            ]
        )
        result = sb["smiling"].mean()
        return result, sb

    sb_dp, sb = sliceby(pivot)

    plot = mk.gui.Plot(
        sb_dp,
        selection=selection,
        x="smiling",
        y="slice",
        x_label="smiling",
        y_label="slice",
        slot="plot",
    )

    @mk.gui.interface_op
    def filter_selection(
        dp: mk.DataPanel, sb, sb_dp: mk.DataPanel, selection: List[int]
    ):
        if len(selection) == 0:
            return dp

        rows = []
        for idx in selection:
            key = sb_dp["slice"][idx]
            rows.extend(sb.slices[key])

        return dp.lz[np.array(rows)]

    filtered_dp = filter_selection(pivot, sb, sb_dp, selection)

    gallery = mk.gui.Gallery(
        dp=filtered_dp,
        main_column="image",
        tag_columns=["smiling"],
        edit_target=mk.gui.EditTarget(
            pivot=pivot, pivot_id_column="image_id", id_column="image_id"
        ),
        slot="gallery",
    )

    # cards = mk.gui.SliceByCards(
    #     sliceby=sb,
    #     main_column="image"
    # )
    # return mk.gui.Interface(components=[cards])
    return mk.gui.Interface(
        layout=mk.gui.Layout("Mocha"), components={"gallery": gallery, "plot": plot}
    )


mk.gui.start()
plot(dp).launch()
