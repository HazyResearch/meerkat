# User Guide

## Python Operators on Stores are reactive
As a convenience we make a select set of Python operators reactive for Stores. This means that if you use these operators on `Store` objects, the result will be a `Store` object that is reactive. For example, the `+` operator is reactive:

```python
x = mk.Store(1)
y = x + 1 

# The `+` operator is reactive.
# y is a Store and will update when x updates.
print("y:", type(y), y.get())
```

A list of reactive operators are listed below:

<!---autogen-start: mk-store-reactive-operators-->
blah
<!---autogen-end: mk-store-reactive-operators-->

asdanasd 

<!---autogen-start: mk-store-reactive-operators-->
blah
<!---autogen-end: mk-store-reactive-operators-->
