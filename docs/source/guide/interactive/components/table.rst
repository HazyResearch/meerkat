:class:`mk.gui.Table`
====================

:class:`mk.gui.Table` is a view-only component that displays the dataframe as a table.

.. collapse:: Code

    .. code-block:: python

        import meerkat as mk

        df = mk.get("imagenette", version="160px")

        table = mk.gui.Table(df=df)

        mk.gui.start(shareable=False)
        mk.gui.Interface(component=mk.gui.RowLayout(components=[table])).launch()

.. raw:: html

    <div style="position: relative; padding-bottom: 62.14039125431531%; height: 0;"><iframe src="https://www.loom.com/embed/afee68dd7e59492eaebdf093077d17d2" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>