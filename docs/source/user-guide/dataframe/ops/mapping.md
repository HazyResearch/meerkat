---
file_format: mystnb
kernelspec:
  name: python3
---

(guide/dataframe/ops/mapping)=
# Mapping: `map` and `defer`

In this guide, we discuss how we can create new columns by applying a function to each row of existing columns: we call this *mapping*. We provide detailed examples of how to the {func}`~meerkat.map` operation. We also introduce the {func}`~meerkat.update` and {func}`~meerkat.filter` operations, which are utilities that wrap the {func}`~meerkat.map` operation. 

```{contents}
:local:
```

## Map

Let's warm up with an example: converting a column of birth years to a column of ages. We start with a small DataFrame of voters with two columns: `birth_year`, which contains the birth year of each person, and `residence`, which contains the state in which each person lives.

```{code-cell} ipython3
:tags: [output_scroll]

import meerkat as mk
df = mk.DataFrame({
    "birth_year": [1967, 1993, 2010, 1985, 2007, 1990, 1943],
    "residence": ["MA", "LA", "NY", "NY", "MA", "MA", "LA"]
})
```

### Map with a single input column
We want to create a new column, `age`, which contains the age of each person. We can do this with the {func}`~meerkat.map` operation, passing a lambda function that takes a birth year and returns the age. 
    
```{code-cell} ipython3
:tags: [output_scroll]

import datetime
df["age"] = df["birth_year"].map(
    lambda x: datetime.datetime.now().year - x
)
df
```
Note that we add the new column to the DataFrame in-place by assigning the result of the {func}`~meerkat.map` operation to the `age` column. 

### Map with multiple input columns 
We can also map a function that takes more than one column as argument. For example, say we wanted to create new column `ma_eligible` that indicates whether or not a person is eligible to vote in Massachusetts. We can do this with the {func}`~meerkat.map` operation, passing a lambda function that takes age and residence.

```{code-cell} ipython3
:tags: [output_scroll]

df["ma_eligible"] = df.map(
    lambda age, residence: (residence == "MA") and (age >= 18)
)
df
```
The call to {func}`~meerkat.map` inspects the signature of the function and determines that it takes two arguments: `age` and `residence`. It then finds the columns with the same names in the DataFrame and passes the corresponding values to the function. If the function takes an non-default argument that is not a column in the DataFrame, the operation will raise a `ValueError`.

We can also specify the correspondence between arguments and columns explicitly with the `inputs` argument to {func}`~meerkat.map`. While the inspection of function signature is convenient, it can be error-prone if used with a function that has a large number of arguments, some of which may spuriously match column names. The cell below is functionally equivalent to the one above, but is slightly more verbose.

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

### Map with multiple output columns 
It's also possible to map a single function that returns multiple values. 

For example, say we wanted to create a two columns `ma_eligible` and `la_eligible` that indicate whether or not a person is eligible to vote in Massachusetts and Louisiana, respectively. We can do this with the {func}`~meerkat.map` operation, passing a lambda function that takes age and residence and returns a tuple of two booleans.

```{code-cell} ipython3
:tags: [output_scroll]

def is_eligibile(age, residence):
    old_enough = age >= 18
    return (residence == "MA") and old_enough, (residence == "LA") and old_enough

df.map(is_eligibile)
```

Note that the output of the function was split into two columns. The names of the columns are just the indices in the tuple returned by the function. We can rename the columns by passing a tuple of column names to the `outputs` argument of {func}`~meerkat.map`. 

```{code-cell} ipython3
:tags: [output_scroll]

df.map(is_eligibile, outputs=("ma_eligible", "la_eligible"))
```

Instead of outputting two columns, one for each state, we may want to output a single {class}`~meerkat.ObjectColumn` containing tuples. To accomplish this we can pass `"single"` to the outputs argument. 

```{code-cell} ipython3
:tags: [output_scroll]

df.map(is_eligibile, outputs="single")
```

```{warning} 
{class}`~meerkat.ObjectColumn` is a column type that can store arbitrary Python objects, but it is backed by a Python list. This means it is **much** slower than other column types. We discuss this more in the guide on {doc}`../column/object`.
```

If the function returns a dictionary, we can skip the `outputs` argument and {func}`~meerkat.map` will automatically use the keys of the dictionary as column names.

```{code-cell} ipython3
:tags: [output_scroll]

def is_eligibile(age, residence):
    old_enough = age >= 18
    return {
        "ma_eligible": (residence == "MA") and old_enough,
        "la_eligible": (residence == "LA") and old_enough
    }

df.map(is_eligibile)
```
If we would like to use a different name for the columns or only use a subset of the keys, we can pass a dictionary to the `outputs` argument. 

```{code-cell} ipython3
:tags: [output_scroll]

df.map(is_eligibile, outputs={"ma_eligible": "ma", "la_eligible": "la"})
```

```{warning}
*Consistent number of outputs.* If a function returns multiple values (either as a tuple or a dictionary), the number of values or the keys must be consistent across all calls to the function. 

*Consistent type of outputs.* It is also important to note that the type of the resulting column is inferred by the first row of the input column(s). As a result, if later rows return values of a different type or shape, an error may be raised because the value cannot be inserted in the inferred column type. To explicitly specify the type of the output column, use the `output_types` argument to {func}`~meerkat.map`.
```


(guide/dataframe/ops/mapping/deferred)=
## Deferred map and chaining
In this section, we discuss how we can chain together multiple map operations using deferred maps. This produces a chain of operations that can be executed together. In the following section, we'll discuss how we can pipeline chained operations to take advantage of parallelism.

```{note}
If you're unfamiliar with DeferredColumns, you may want to read the guide on {doc}`../column/deferred` before diving into this section. 
```

### Simple deferred map
 
In addition to {func}`~meerkat.map` described above, Meerkat provides {func}`~meerkat.defer`, which creates a {class}`~meerkat.DeferredColumn` representing a deferred map. The two functions share nearly the exact same signature (*i.e.* all that was discussed in the previous section around multiple inputs and ouputs also applies to {func}`~meerkat.defer`). The difference is that {func}`~meerkat.defer` returns a column that has not yet been computed. It is a placeholder for a column that will be computed later.

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

### Chaining deferred maps

Let's motivate the use of deferred maps with a more involved example: processing a dataset of images. We're going to use the [Imagenette dataset](https://github.com/fastai/imagenette#image%E7%BD%91), a small subset of the original [ImageNet](https://www.image-net.org/update-mar-11-2021.php).   We can load it from the Meerkat dataset registry with the {func}`~meerkat.get` function. This dataset is made up of 10 classes (e.g. "garbage truck", "gas pump", "golf ball"), but we'll focus on a simpler binary classification task: "parachute" vs. "golf ball".

```{code-cell} ipython3
:tags: [output_scroll]

df = mk.get("imagenette")
df = df[df["label"].isin(["parachute", "golf ball"])].sample(500)
df[["img", "label", "path"]]
```

Below, we've defined a classifier with a silly decision rule: it classifies an image as containing a parachute if in more than half of the pixels, the **blue** channel has the highest value (the logic being that parachutes are often photographed in the sky). 

The classifier has two methods which we need to chain together: `preprocess` and `predict`. The `preprocess` method takes an image and returns a NumPy array of shape `(224, 224, 3)`. The `predict` method takes a batch of images and returns a boolean array with predictions.

```{margin} To batch or not to batch?
Note that `preprocess` takes a single image, while `predict` takes a batch of images. This is a common pattern in machine learning: preprocessing functions are often singleton while predict functions are often batched. Consider, for example, the `preprocess` and `embed_image` functions in the implementation of [CLIP](https://github.com/openai/CLIP), a popular image encoder.
```

```{code-cell} ipython3
from PIL.Image import Image
import numpy as np

class ParachuteClassifier:
    
    def preprocess(self, img: Image) -> np.ndarray:
        """Prepare an image for classification."""
        return np.array(img.convert("RGB").resize((224, 224)))
    
    def predict(self, batch: np.ndarray) -> np.ndarray:
        """Classify a batch of images as containing a parachute or not, using a 
        simple decision rule. 
        """
        return (np.argmax(batch, axis=3) == 2).mean(axis=1).mean(axis=1) > 0.5

classifier = ParachuteClassifier()
```
Because only one of the two methods is batched, chaining them correctly is a bit tricky (one approach might invovle a double for-loop). Fortunately, with deferred maps, we can chain together functions that use different batching strategies. First, we create a deferred column that applies `preprocess` to each image – by default `is_batched_fn=False`, so the `preprocess` method is applied to each image individually. Next, we map the `predict` method over the deferred column. Because `predict` is batched, we can use `is_batched_fn=True` and specify a `batch_size` to indicate that the `predict` method should be applied to batches of images.

```{code-cell} ipython3
preprocessed = df["img"].defer(classifier.preprocess)
df["prediction"] = preprocessed.map(
    classifier.predict, is_batched_fn=True, batch_size=32
)
```

Finally we can compute the accuracy of the classifier by mapping a simple function that compares the predictions to the ground truth labels. 

```{code-cell} ipython3
accuracy = df.map(lambda prediction, label: prediction == (label == "parachute")).mean()
print(f"Accuracy: {accuracy:.2%}")
``` 
Not too bad! I guess you can get pretty far in machine learning relying on spurious correlations. 

We could have chained all of these operations together into a single chain. 
```{code-cell} ipython3
preprocessed = df["img"].defer(classifier.preprocess)
df["prediction"] = preprocessed.defer(
    classifier.predict, is_batched_fn=True, batch_size=32
)
accuracy = df.map(lambda prediction, label: prediction == (label == "parachute")).mean()
```

Here's a trick question: *How long is the resulting chain?* At first glance, it seems like the chain is three maps long: `preprocess`, `predict`, then `accuracy`. However, recall from the guide on {doc}`../column/deferred` that images and other complex data types are typically stored in {class}`~meerkat.DeferredColumn`s – that is, the `"img"` column is itself a deferred map. This map applies an image loading function to each filepath in the dataset. It could have been created with a line like `df["img"] = df["filepath"].defer(load_image)`. Because our first {func}`~meerkat.defer` call in the cell above was made on the `"img"` column, the resulting chain is actually four maps long: `load`, `preprocess`, `predict`, then `accuracy`. 

```{figure} _figures/map_chain.png
---
name: map-chain
---
The chain of maps created by the code above. Although we only called {func}`~meerkat.defer` or {func}`~meerkat.map` three times, the resulting chain is four maps long because the `"img"` column is itself a deferred map.
% Google drawing: https://docs.google.com/drawings/d/13NQT5B54-RMItlPezt2AJwl4tZg4P4X46ujxkCoXWKE/edit?usp=sharing
```


*Why not just use multiple `map` calls?* Chaining together deferred maps has two main advantages over simply calling `map` multiple times. 
1. **Memory.** If one of the intermediate maps produces images or other large data types (as does `preprocess` in this example), then the resulting column may not be able to fit in memory. With a regular `map`, that entire column will be materialized before the next `map` begins. If we use a deferred map, then intermediate results are released from memory once they are consumed by the next map in the chain. This enables us to process data types that are too large to fit in memory (*e.g.* images, video, audio).
2. **Parallelism.** Using deferred maps allows us to better take advantage of the parallelism afforded by our system, especially if each map in the chain depends on different system resources (*e.g.* CPU vs. GPU vs. I/O bandwidth). Meerkat supports pipelining chained maps, which allows us to run multiple maps in parallel. We discuss this in the next section. 

## Pipelining and Parallelism 

Because map applies the same function to each row, it is a [delightfully parallelizable](https://en.wikipedia.org/wiki/Embarrassingly_parallel) operation. In this section, we discuss how to parallelize maps and how to pipeline a chain of parallel maps. 

```{danger} WIP
Pipelining is currently an experimental feature. It will be implemented using [Ray](https://docs.ray.io/en/latest/data/pipelining-compute.html). 
```









