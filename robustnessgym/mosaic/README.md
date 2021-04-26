<div align="center">
    <img src="../../docs/mosaic.png" height=100 alt="Mosaic logo"/>
</div>

-----

The Mosaic project's goal is to create clean abstractions for ML practitioners to load, manipulate, train, evaluate, inspect, visualize and interact with complex high-dimensional, multi-modal data. 

Mosaic's core contribution is the `DataPane`, a simple columnar data abstraction that can house arbitrary columns of complex data side-by-side. The `DataPane` allows users to work with text, image, medical imaging, time-series, video, and other complex Python objects with clean, high-level interfaces.


## `DataPane`

A `DataPane` is a Python object that contains a collection of columns. 

```python
from robustnessgym.mosaic import DataPane

# Create a simple DataPane
dp = DataPane()
```