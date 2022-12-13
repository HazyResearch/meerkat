
Overview of Column Types
=========================

There are four core column types in Meerkat, each with a different interface.

1. :class:`~meerkat.ScalarColumn` Each row stores a single numeric or string value. These columns have an interface similar to a Pandas Series. 
2. :class:`~meerkat.TensorColumn` Each row stores an identically shaped multi-dimensional array (*e.g.* vector, matrix, or tensor). These columns have an interface similar to a NumPy ndarray. 
3. :class:`~meerkat.ObjectColumn` Each row stores an arbitrary Python object. These columns should be used sparingly, as they are significantly slower than the columns above. However, they may be useful in small DataFrames. 
4. :class:`~meerkat.DeferredColumn` Represents a *deferred* map operations. A DeferredColumn maintains a single function and a pointer to another column. Each row represents (*but does not actually store*) the value returned from applying the function to the corresponding row of the other column.

.. admonition:: Flexibility in Implementation

    Meerkat columns are simple wrappers around well-optimized data structures from other libraries. These libraries (e.g. NumPy) run compiled machine code that is significantly faster than routines written in Python. 

    The data structure underlying a column is available through the ``.data`` attribute of the column. For example, the following code creates a :class:`~meerkat.TensorColumn` and then accesses the underlying NumPy array.

    .. ipython:: python

        import meerkat as mk;
        col = mk.TensorColumn([0,1,2]);
        col.data


    Meerkat is unopinionated when it comes to the choice of data structure underlying columns. This provides users the **flexibility** to choose the best data structure for their use case.
    For example, a `TensorColumn` can be backed by either a `NumPy Array  <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_) or a `PyTorch Tensor <https://pytorch.org/docs/stable/tensors.html>`_. 
    
    Each ``ScalarColumn`` object in Meerkat is actually an instance of one of its subclasses (:class:`~meerkat.PandasScalarColumn`, :class:`~meerkat.ArrowScalarColumn`). These subclasses are responsible for implementing the :class:`ScalarColumn` interface for a choice of data structure. Similarly, each ``TensorColumn`` object is an instance of its subclasses (:class:`~meerkat.NumPyTensorColumn`, :class:`~meerkat.TorchTensorColumn`). 

    *How to pick a subclass?* In general, users should not have to think about which subclass to use. Meerkat chooses a subclass based on the data structure of the input data. For example, the following code creates a ``ScalarColumn`` backed by a Pandas Series:

    .. ipython:: python

        mk.column([0,1,2])

    You can also explicitly specify the subclass to use. For example, the following code creates a ``ScalarColumn`` backed by an Arrow array:

    .. ipython:: python

        mk.ArrowScalarColumn([0,1,2])