import meerkat as mk

df = mk.get("imagenette")
df = df.sample(n=1000, replace=False)
# df = mk.embed(df, input="img")


def match(df: mk.DataFrame, against: str, id_column: str) -> mk.gui.Interface:

    # Setup pivots
    df_pivot = mk.gui.Pivot(df)

    # Setup components
    match = mk.gui.Match(df_pivot, against=against, col=id_column)

    sort_derived = mk.sort(df_pivot, by=match.col, ascending=False)

    gallery = mk.gui.Gallery(
        sort_derived,
        main_column="img",
        tag_columns=["label"],
        edit_target=mk.gui.EditTarget(df_pivot, id_column, id_column),
    )
    return mk.gui.Interface(components=[match, gallery])


mk.gui.start()
match(df, against="img", id_column="path").launch()
