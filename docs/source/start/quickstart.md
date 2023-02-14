---
file_format: mystnb
kernelspec:
  name: python3
---


# Quickstart

We'll do a whirlwind tour of the basic features of Meerkat.
Make sure you have Meerkat [installed and running](./install) before you
go through this quickstart.

In this quickstart we will create a simple dashboard that plots the label distribution
on a filtered subset of the `imagenette` dataset, a tiny subset of ImageNet.

We're going to

* Load in `imagenette` into a Meerkat DataFrame
* Walk through how to create a dashboard
* Run the dashboard

Let's start!


## Loading in the data

We'll start by loading in the `imagenette` dataset into a Meerkat DataFrame.
Meerkat includes a few datasets that you can use to get started, including `imagenette`.

```{code-cell} python
import meerkat as mk

# Load in the imagenette dataset
df = mk.get('imagenette', version="160px")

# Prints out the first 5 rows of the DataFrame
df.head(5)
```

## Creating a dashboard

Now that we have our data loaded in, let's create a dashboard that plots the label distribution
on a filtered subset of the `imagenette` dataset.

```{code-cell} python
mk.gui.start()
```

```{code-cell} python
df.gui.table()
```

