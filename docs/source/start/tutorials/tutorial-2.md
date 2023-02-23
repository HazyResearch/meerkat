# Tutorial 2: Reactive Image Viewer

In this tutorial, we will build a simple image viewer that shows a random subset of images from
a class in an image dataset.

Through this tutorial, you will learn about:
- the concept of **reactive functions** in Meerkat
- how chaining reactive functions together can be used to build complex applications
- a few more Meerkat components that you can use in your applications

To get started, you can run the tutorial demo script that we provide.
```{code-block} bash
mk demo tutorial-2
```

```python
import meerkat as mk

df = mk.get("imagenette", version="160px")
IMAGE_COL = "img"
LABEL_COL = "label"
```