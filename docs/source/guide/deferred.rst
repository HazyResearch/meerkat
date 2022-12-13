
Deferred Columns
=================

*Motivation.* When working with multimodal datasets, the data in some columns may fit easily in memory, while the data in others are best kept on disk and loaded only when needed. For example, in an image dataset, the image labels and metadata are small and may fit in memory, while the images themselves are large and should stay on disk until they are needed.

In Meerkat, columns like :class:`~meerkat.ImageColumn` and :class:`~meerkat.AudioColumn` make it easy to work with complex data types that can't fit in memory. If you check out the implementation of these classes, you'll notice that they are straightforward subclasses of :class:`~meerkat.DeferredColumn`.  

*What's a DeferredColumn?* A  :class:`~meerkat.DeferredColumn` wraps around another column and *represents* what you would get if you applied a function to its content. You can think of it as a deferred map operation. 

Consider the following example, where we create a simple Meerkat column...    

.. ipython:: python

    import meerkat as mk

    col = mk.column(list(range(10)))

  
...and create a deferred column, ``dcol``, based on it:

.. ipython:: python

    dcol = col.defer(function=lambda x: x + 10)
    dcol

Like other columns, deferred columns can be subselected.

.. ipython:: python

    small_dcol = dcol[:5]

Unlike other columns, deferred columns are **callable**. When we call a deferred column, we apply the function to the underlying column.

.. ipython:: python

    small_dcol()

Critically, the function inside a deferred column is called neither on creation or selection, but only later once the column is called! This is very useful for columns with large data types that we don't want to load all into memory at once. For example, we could create a :class:`~meerkat.DeferredColumn` that lazily loads images...

.. ipython:: python
    :verbatim:

    from PIL import Image
    
    df = mk.DataFrame(
        {
            "filepath": ["/abs/path/to/image0.jpg", ...], 
            "image_id": ["image0", ...] 
        }
    )
    df["image"] = df["filepath"].defer(fn=Image.open)

Notice how we provide an absolute path to the images. This makes the column usable from any working directory. 
However, using absolute paths is in other ways not ideal: what if we want to share the DataFrame and open it on a different machine? In the section below, we discuss a subclass of :class:`~meerkat.DeferredColumn` that makes it easy to manage filepaths. 

FileColumn
########### 

As discussed above, :class:`~meerkat.DeferredColumn`s are commonly used to load files from disk. To make it easier to work with file loading columns, Meerkat provides the :class:`~meerkat.FileColumn`, a simple subclass of :class:`~meerkat.DeferredColumn`. 

The :class:`~meerkat.FileColumn` constructor takes an additional argument, ``base_dir``, which is the base directory from which all file paths are relative. 
When ``base_dir`` is provided, the paths passed to ``filepaths`` should be relative to ``base_dir``:

.. ipython:: python
    :verbatim:

    from PIL import Image

    df = mk.DataFrame(
        {
            "filepath": ["image0.jpg", ...], 
            "image_id": ["image0", ...] 
        }
    )
    df["image"] = mk.FileColumn.from_filepaths(
        filepaths=df["filepath"],
        loader=Image.open,
        base_dir="/abs/path/to",
    )


The ``base_dir`` can then be changed at any time, so if we wanted to share the DataFrame with another user, we could instruct them to reset the base_dir using ``df["image"].base_dir = "/other/users/abs/path/to"``. Introducing this additional step isn't ideal though, so we recommend using the environment variables technique as described below.

.. admonition:: Using Environment Variables in ``base_dir``

    Environment variables in the ``base_dir`` argument are automatically expanded. For example, if you set the environment variable ``MEERKAT_BASE_DIR`` to ``"/abs/path/to"``, then you can use ``df["image"].base_dir = "$MEERKAT_BASE_DIR/path/to"``. This is ideal for sharing DataFrames between different users and machines. 

    Note that the Meerkat dataset registry relies heavily on this technique, using a special environment variable ``MEERKAT_DATASET_DIR`` that points to the ``mk.config.datasets.root_dir``. 
    

An :class:`~meerkat.ImageColumn` is a just a :class:`~meerkat.FileColumn` like this one, with a few more bells and whistles!


Chaining Deferred Columns
##########################

.. todo::

    Fill in this stub.
