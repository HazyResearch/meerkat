:class:`mk.gui.Sort`
=====================

:class:`mk.gui.Sort` allows the user to sort the data by different columns (i.e. criteria).

Users can dynamically 

1. add/remove sorting criteria
2. rearrange criteria to change the order sorting is app
3. change sorting to be ascending/descending for each criterion

Example
-------

.. collapse:: Code

    .. code-block:: python

        import meerkat as mk

        df = mk.get("imagenette", version="160px")

        with mk.reactive():
            filter = mk.gui.Filter(df=df)
            df = filter(df)

            criteria = mk.Store([])
            sort = mk.gui.Sort(df=df, criteria=criteria)
            df = sort(df)

        # Gallery
        gallery = mk.gui.Gallery(
            df=df,
            main_column="img",
        )

        mk.gui.start(shareable=False)
        mk.gui.Interface(
            component=mk.gui.RowLayout(components=[sort, gallery])
        ).launch()

