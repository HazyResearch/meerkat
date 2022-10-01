import meerkat as mk

dp = mk.get("imagenette")
dp = dp.sample(n=1000, replace=False)
# dp = mk.embed(dp, input="img")


def match(dp: mk.DataPanel, against: str, id_column: str) -> mk.gui.Interface:

    # Setup pivots
    dp_pivot = mk.gui.Pivot(dp)

    # Setup components
    match = mk.gui.Match(dp_pivot, against=against, col=id_column)

    sort_derived = mk.sort(dp_pivot, by=match.col, ascending=False)

    gallery = mk.gui.Gallery(
        sort_derived,
        main_column="img",
        tag_columns=["label"],
        edit_target=mk.gui.EditTarget(dp_pivot, id_column, id_column),
    )
    return mk.gui.Interface(components=[match, gallery])


mk.gui.start()
match(dp, against="img", id_column="path").launch()
