
Overview of Column Types
=========================

There are four core column types in Meerkat, each with a different interface.

1. :class:`~meerkat.ScalarColumn` Each row stores a single numeric or string value. These columns have an interface similar to a Pandas Series. 
2. :class:`~meerkat.TensorColumn` Each row stores an identically shaped multi-dimensional array (*e.g.* vector, matrix, or tensor). These columns have an interface similar to a NumPy ndarray. 
3. :class:`~meerkat.ObjectColumn` Each row stores an arbitrary Python object. These columns should be used sparingly, as they are significantly slower than the columns above. However, they may be useful in smaller DataFrames. 
4. :class:`~meerkat.LambdaColumn` Maintains a single function and a pointer to another column. Each row represents (*but does not actually store*) the value returned from applying the function to the corresponding row of the other column. Importantly, these columns are **callable**. 

.. admonition:: Underlying Data Structures

    Meerkat columns simply wrap other popular data structures. For example, a `TensorColumn` is a wrapper around a NumPy n-dimensional array. 

    Meerkat aims to be as unopinionated as possible when it comes to the choice of data structure. So, we also provide subclasses of ``ScalarColumn`` and ``TensorColumn`` that use different underlying data structures. For example, we provide :class:`ArrowScalarColumn` (a column type with almost the same interface as `ScalarColumn`, but implemented with Apache Arrow) and :class:`~meerkat.TorchTensorColumn` (a column type with almost the same interface as :class)








