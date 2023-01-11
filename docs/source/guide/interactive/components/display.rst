Displaying dataframes
=====================

:class:`mk.gui.Gallery`
-----------------------

:class:`mk.gui.Gallery` is is view-only component that allows you to visualize your dataframe in a gallery.
You can visualize your data along with your metadata for fast visual inspection.

Each example displayed in the gallery can be expanded as a modal by double clicking on the example.
The modal displays the data and metadata for the example.


:class:`mk.gui.Table`
---------------------

:class:`mk.gui.Table` is also a view-only component that displays the dataframe as a table.

Example
-------
Let's look at an example where we can switch between the gallery and table views.

.. collapse:: Code

    .. code-block:: python

        import meerkat as mk

        df = mk.get("imagenette", version="160px")


        table = mk.gui.Table(df=df)
        gallery = mk.gui.Gallery(df=df, main_column="img")
        tabs = mk.gui.Tabs(tabs={"Table": table, "Gallery": gallery})

        mk.gui.start(shareable=False)
        mk.gui.Interface(component=mk.gui.RowLayout(components=[tabs])).launch()


.. raw:: html

   <div style="position: relative; padding-bottom: 62.14039125431531%; height: 0;"><iframe src="https://www.loom.com/embed/c4fe548a57244cc1bb760342ed61192d" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>