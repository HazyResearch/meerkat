
Lambda Columns and Lazy Selection
==================================

Lambda Columns
--------------

If you check out the implementation of :class:`~meerkat.ImageColumn`, you'll notice that it's a super simple subclass of :class:`~meerkat.LambdaColumn`. 

*What's a LambdaColumn?* In Meerkat, high-dimensional data types like images and videos are typically stored in a :class:`~meerkat.LambdaColumn`. A  :class:`~meerkat.LambdaColumn` wraps around another column and applies a function to it's content as it is indexed. 

Consider the following example, where we create a simple Meerkat column...    

.. ipython:: python

    import meerkat as mk

    col = mk.NumpyArrayColumn([0,1,2])
    col[1]

  
...and wrap it in a lambda column.

.. ipython:: python

    lambda_col = col.to_lambda(function=lambda x: x + 10)
    lambda_col[1]  # the function is only called at this point!


Critically, the function inside a lambda column is only called at the time the column is indexed! This is very useful for columns with large data types that we don't want to load all into memory at once. For example, we could create a :class:`~meerkat.LambdaColumn` that lazily loads images...

.. ipython:: python
    :verbatim:

    from PIL import Image
    
    df = mk.DataFrame(
        {
            "filepath": ["/abs/path/to/image0.jpg", ...], 
            "image_id": ["image0", ...] 
        }
    )
    df["image"] = df["filepath"].to_lambda(fn=Image.open)

Notice how we provide an absolute path to the images. This makes the column useable from any working directory. 
However, using absolute paths is in other ways not ideal: what if we want to share the DataFrame and open it on a different machine? In the section below, we discuss a subclass of :class:`~meerkat.LambdaColumn` that makes it easy to manage filepaths. 

FileColumn
########### 

As discussed above, :class:`~meerkat.LambdaColumn`s are commonly used to load files from disk. To make it easier to work with file loading columns, Meerkat provides the :class:`~meerkat.FileColumn`, a simple subclass of :class:`~meerkat.LambdaColumn`. 

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

Lazy Selection
--------------

.. todo::

    Fill in this stub.