---
file_format: mystnb
kernelspec:
  name: python3
---

(guide/test)=

# Hello

This is a test file. 


```{code-cell} ipython3
import meerkat as mk
mk.get("imagenette")[["label", "img"]]
```