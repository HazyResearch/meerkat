
import meerkat as mk

block_match():

    match(block.stuff)
    # update state



df = mk.get("rfw")


table = df.gui.table()

match = mk.gui.match(df)


with Block(df=blah) as block:
    # All children will reflect what the current `df` variable's data is!
    block.add(Match())

with Block(df=blah, static=True) as block:
    # df is `const` and cannot be updated for this block`
    block.add(
        Table()
    )

    block.operators['match']

mk.launch(
    [
        match, 
        table,

    ]
) 