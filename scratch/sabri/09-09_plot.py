from typing import List

import numpy as np

import meerkat as mk

df = mk.get("celeba")


def plot(df: mk.DataFrame) -> mk.gui.Interface:
    pivot = mk.gui.Reference(df)
    selection = mk.gui.Store([])

    @mk.gui.reactive
    def sliceby(df: mk.DataFrame):
        sb = df.sliceby(
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

    sb_df, sb = sliceby(pivot)

    plot = mk.gui.Plot(
        sb_df,
        selection=selection,
        x="smiling",
        y="slice",
        x_label="smiling",
        y_label="slice",
        slot="plot",
    )

    @mk.gui.reactive
    def filter_selection(
        df: mk.DataFrame, sb, sb_df: mk.DataFrame, selection: List[int]
    ):
        if len(selection) == 0:
            return df

        rows = []
        for idx in selection:
            key = sb_df["slice"][idx]
            rows.extend(sb.slices[key])

        return df.lz[np.array(rows)]

    filtered_df = filter_selection(pivot, sb, sb_df, selection)

    gallery = mk.gui.Gallery(
        df=filtered_df,
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
plot(df).launch()
