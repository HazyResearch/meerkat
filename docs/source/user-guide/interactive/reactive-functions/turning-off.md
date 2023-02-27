# Turning Off Reactivity

In some cases, we may not want certain code to be reactive. To make code non-reactive, we can use {class}`meerkat.unmarked`.

```{important}
Use `unmarked` to turn off reactivity for a function or block of code.
```

Like `reactive`, `unmarked` can be used as a decorator for functions:

```python
@unmarked()
def add(a, b):
    # This will never return a store because the function is unmarked.
    return a + b


x = mk.Store(1)
y = mk.Store(2)
c = add(x, y)  # c is 3, type(c) is int
```

It can also be used as a context manager:

```python
def add(a, b):
    # This will never return a store because the function is unmarked (by default).
    return a + b


x = mk.Store(1)
y = mk.Store(2)

with unmarked():
    c = add(x, y)  # c is 3, type(c) is int
```

If `reactive` functions are run in an unmarked context, they will not be reactive:

```python
@reactive()
def add(a, b):
    return a + b


x = mk.Store(1)
y = mk.Store(2)

with unmarked():
    c = add(x, y)  # c is 3, type(c) is int
```
