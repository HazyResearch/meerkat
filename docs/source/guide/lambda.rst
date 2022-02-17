
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

    lambda_col = col.to_lambda(fn=lambda x: x + 10)
    lambda_col[1]  # the function is only called at this point!


Critically, the function inside a lambda column is only called at the time the column is indexed! This is very useful for columns with large data types that we don't want to load all into memory at once. For example, we could create a :class:`~meerkat.LambdaColumn` that lazily loads images...

.. ipython:: python
    :verbatim:
    
    filepath_col = mk.PandasSeriesColumn(["path/to/image0.jpg", ...])
    img_col = filepath_col.to_lambda(fn=load_image)


An :class:`~meerkat.ImageColumn` is a just a :class:`~meerkat.LambdaColumn` like this one, with a few more bells and whistles!

Lazy Selection
--------------

.. todo::

    Fill in this stub.