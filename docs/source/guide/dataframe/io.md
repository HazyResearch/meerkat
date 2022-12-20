---
file_format: mystnb
kernelspec:
  name: python3
---

(guide/dataframe/selection)=

# I/O

In this guide, we will discuss how to bring data into Meerkat from various file formats and other Python libraries. 
We will also discuss how to export data in Meerkat back into these formats. Finally, we'll also discuss how to persist Meerkat DataFrames to disk

## Importing into Meerkat

Meerkat has a number of built-in functions for reading in data from various file formats and Python libraries. 
We'll provide one in depth example for reading in data from a CSV file, and then provide a list of the other supported file formats and libraries.

### *Example*: Importing a dataset from CSV

Let's load a CSV file from disk and read it into a Meerkat DataFrame. 
We will be using a small sample of data from the [National Gallery of Art Open Data Program](https://github.com/NationalGalleryOfArt/opendata). We've included this data at `_data/art_ngoa.csv` in the Meerkat repository.

We will use the {func}`~meerkat.from_csv` function to read in the data.

```{code-cell} ipython3
import meerkat as mk

df = mk.from_csv("_data/art_ngoa.csv")
df.head()
```

 Notice that each row corresponds to a single work of art, and each column corresponds to a different attribute of the work of art. 

 **Representing images.** The last column, `iiifthumburl`, contains a URL to a thumbnail image of the work.
Using {func}`~meerkat.image`, we can download the thumbnail image and display it in the DataFrame.


```{code-cell} ipython3
df["image"] = mk.image(df["iiifthumburl"], downloader="url")
df[["title", "attribution", "image"]].head()
```
The function `mk.image` creates a {class}`~meerkat.ImageColumn` which defers the downloading of images from the URLs until the data is needed. 

```{admonition} Deferred Columns
If you're wondering how {class}`~meerkat.ImageColumn` works, check out the guide on {doc}`columns/deferred`. 
```

**Adding a primary key**.The `objectid` column contains a unique identifier for each work of art. We can use {func}`~meerkat.set_primary_key` to set this column as the primary key for the DataFrame, which allows us to perform key-based indexing on the DataFrame.

```{code-cell} ipython3
df = df.set_primary_key("objectid")
df.loc[221224]
```
The {func}`~meerkat.from_csv` function has a utility parameter `primary_key` which can be used to set the primary key when the DataFrame is created. 
```{code-cell} ipython3
df = mk.from_csv("_data/art_ngoa.csv", primary_key="objectid")
```
```{admonition} Primary Keys
To learn more about primary keys and key-based indexing, check out the section {ref}`key-based-selection`. 
```

### Importing from storage formats
Meerkat supports importing data from a number of other file formats. As in the example above, you may need to set the primary key and/or add additional columns for complex data (*e.g.* images, audio).

- {func}`~meerkat.from_csv()`: Reads in data from a CSV (comma-separated values) file. CSV files are a common format for storing tabular data.

- {func}`~meerkat.from_feather()`: Reads in data from a [Feather file](https://arrow.apache.org/docs/python/feather.html). Feather is a language-agnostic file format for storing DataFrames. It can provide significantly faster I/O than CSV. 

- {func}`~meerkat.from_parquet()`: Reads in data from a [Parquet file](https://parquet.apache.org/). Parquet is a columnar storage format that is designed for efficiency. 

### Importing from other libraries 
It's also posible to import data from third-party Python libraries like [Pandas](https://pandas.pydata.org/) and [HuggingFace Datasets](https://huggingface.co/datasets).



## Exporting from Meerkat


## Writing Meerkat DataFrames to disk


## Reading Meerkat DataFrames from disk 



