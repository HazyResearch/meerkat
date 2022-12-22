---
file_format: mystnb
kernelspec:
  name: python3
---

(guide/dataframe/ops/mapping)=
# Mapping

In this guide, we discuss how we can create new columns by applying a function to each row of existing columns: we call this *mapping*. We provide detailed examples of how to the {func}`~meerkat.map` operation. We also introduce the {func}`~meerkat.update` and {func}`~meerkat.filter` operations, which are utilities that wrap the {func}`~meerkat.map` operation. 

## Map

Let's warm up with an example: converting a column of birth years to a column of ages. We start with a small DataFrame of voters with two columns: `birth_year`, which contains the birth year of each person, and `voter`, which indicates whether or not they voted in the last election.

```{code-cell} ipython3
:tags: [output_scroll]

import meerkat as mk
df = mk.DataFrame({
    "birth_year": [1967, 1993, 2010, 1985, 2007, 1990, 1943],
    "residence": ["MA", "LA", "NY", "NY", "MA", "MA", "LA"]
})
```

**Mapping a single column.** We want to create a new column, `age`, which contains the age of each person. We can do this with the {func}`~meerkat.map` operation, passing a lambda function that takes a birth year and returns the age. 
    
```{code-cell} ipython3
:tags: [output_scroll]

import datetime
df["age"] = df["birth_year"].map(
    lambda x: datetime.datetime.now().year - x
)
df
```
Note that we add the new column to the DataFrame in-place by assigning the result of the {func}`~meerkat.map` operation to the `age` column. 

**Mapping multiple columns.** We can also map a function that takes more than one column as argument. For example, say we wanted to create new column `ma_eligible` that indicates whether or not a person is eligible to vote in Massachusetts. We can do this with the {func}`~meerkat.map` operation, passing a lambda function that takes age and residence.

```{code-cell} ipython3
:tags: [output_scroll]

df["ma_eligible"] = df.map(
    lambda age, residence: (residence == "MA") and (age >= 18)
)
df
```
The call to {func}`~meerkat.map` inspects the signature of the function and determines that it takes two arguments: `age` and `residence`. It then finds the columns with the same names in the DataFrame and passes the corresponding values to the function. If the function takes an non-default argument that is not a column in the DataFrame, the operation will raise a `ValueError`.

We can also specify the correspondance between arguments and columns explicitly with the `inputs` argument to {func}`~meerkat.map`. While the inspection of function signature is convenient, it can be error-prone if used with a function that has a large number of arguments, some of which may spuriously match column names. The cell below is functionally equivalent to the one above, but is slightly more verbose.

```{code-cell} ipython3
:tags: [output_scroll]

df["ma_eligible"] = df.map(
    lambda x, y: (x == "MA") and (y >= 18),
    inputs={"age": "y", "residence": "x"}
)
df
```

```{note}
Some readers may wonder whether `func`{map} was the right choice
in the above example. After all, we could have written a vectorized expression `df["ma_eligible"] = (df["residence"] == "MA") & (df["age"] >= 18)`. In many cases, this would indeed be more efficient. The example above is meant to illustrate the general pattern of using {func}`~meerkat.map`. The examples in the following sections will highlight the benefits of {func}`~meerkat.map`.
```


## Deferred maps and chaining
In this section, we discuss how we can chain together multiple map operations using deferred maps. This produces a chain of operations that can be executed together. In the following section, we'll discuss how we can pipeline chained operations to take advantage of parallelism.

```{note}
If you're unfamiliar with DeferredColumns, you may want to read the guide on {doc}`guide/dataframe/columns/deferred` before diving into this section. 
```

**Deferred maps**. In addition to {func}`~meerkat.map` described above, Meerkat provides {func}`~meerkat.defer`, which creates a {class}`~meerkat.DeferredColumn` representing a deferred map. The two functions share nearly the exact same signature, the difference is that {func}`~meerkat.defer` returns a column that has not yet been computed. It is a placeholder for a column that will be computed later.

To demonstrate, let's repeat the example above, this time using a deferred map to create a 2-step chain of operations. 


```{code-cell} ipython3
# no computation yet
df["age"] = df["birth_year"].defer(
    lambda x: datetime.datetime.now().year - x
)

# computation is done here
df["ma_eligible"] = df.map(
    lambda age, residence: (residence == "MA") and (age >= 18)
)
df
```

The only difference between this code and the code in the previous section is that here we use {func}`~meerkat.defer` instead of {func}`~meerkat.map` when creating the `age` column. The result is the same, but here the computation of both `"age"` and `"ma_eligible"` is performed together at the end, instead of in two stages. 


**A more involved example.** Let's motivate the use of deferred maps with a more involved example: processing a dataset of images. We're going to use the [Imagenette dataset](https://github.com/fastai/imagenette#image%E7%BD%91), a small subset of the original [ImageNet](https://www.image-net.org/update-mar-11-2021.php).  This dataset is made up of 10 classes (e.g. "garbage truck", "gas pump", "golf ball"). We can load it from the Meerkat dataset registry with the {func}`~meerkat.get` function. 

```{code-cell} ipython3
:tags: [output_scroll]

df = mk.get("imagenette").sample(100)
df.head()
```

We'd like to apply the following 


```{code-cell} ipython3
from PIL.Image import Image
import numpy as np

class ParachuteClassifier:
    
    def preprocess(self, img: Image) -> np.ndarray:
        """Prepare an image for classification."""
        return np.array(img.convert("RGB"))
    
    def predict(self, batch: np.ndarray) -> np.ndarray:
        """Classify a batch of images as containing a parachute or not."""
        print("here", batch.shape, batch.data)
        return batch[:, :, :, 2].mean(axis=1).mean(axis=1) > 0.5

classifier = ParachuteClassifier()
```

```{code-cell} ipython3
preprocessed = df["img"].defer(classifier.preprocess)
df["prediction"] = preprocessed.map(
    classifier.predict, is_batched_fn=True, batch_size=32
)
```

```{code-cell} ipython3
accuracy = df.map(lambda prediction, label: prediction == (label == "parachute")).mean()
accuracy
```




## Pipelining and Parallelism 









### Pipelining Computation 







