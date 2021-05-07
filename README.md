<div align="center">
    <img src="docs/mosaic.png" height=100 alt="Mosaic logo"/>
</div>

-----

The Mosaic project's goal is to create clean abstractions for ML practitioners to load, manipulate, train, evaluate, inspect, visualize and interact with complex high-dimensional, multi-modal data. 

Mosaic's core contribution is the `DataPane`, a simple columnar data abstraction that can house arbitrary columns of complex data side-by-side. The `DataPane` allows users to work with text, image, medical imaging, time-series, video, and other complex Python objects with clean, high-level interfaces.


## `DataPane`

A `DataPane` is a Python object that contains a collection of columns. 


#### Create a `DataPane`
```python
from mosaic import DataPane

# Create a simple DataPane
dp = DataPane({
    'text': ['The quick brown fox.', 'Jumped over.', 'The lazy dog.'],
    'imagepath': ['fox.png', 'jump.png', 'dog.png'],
})
```


## `Column`

Mosaic supports a variety of columns.

```python
from mosaic import SomeColumn

sc = SomeColumn(data=some_data)
len(sc) == len(some_data)

```



## `Cell`
