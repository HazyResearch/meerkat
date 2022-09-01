
import meerkat as mk

block_match():

    match(block.stuff)
    # update state



dp = mk.get("rfw")


table = dp.gui.table()

match = mk.gui.match(dp)


with Block(dp=blah) as block:
    # All children will reflect what the current `dp` variable's data is!
    block.add(Match())

with Block(dp=blah, static=True) as block:
    # dp is `const` and cannot be updated for this block`
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