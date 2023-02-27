(guide_store_faq)=

# FAQs

## How do I create a `Store`?

```python
x = mk.Store(1) # or mk.Store(<any object>)
```

## What can I put in a `Store`?

_Anything._ All Python objects are valid inputs to `Store`. For example, primitive types (e.g. `int`, `float`, `str`, etc.), complex types (e.g. `list`, `dict`, `tuple`, etc.), arbitrary objects (e.g. `pandas.DataFrame`, `numpy.ndarray`, etc.), classes, functions and even other `Store` objects (although we don't recommend nesting `Store` objects).

## Will this `<statement>` be reactive, and return a `Store`?

Normally, statements with `Store` objects are reactive _only_ when they involve operators. For example, `x + y`, `x - 1` and `~x` will all behave as reactive statements that return another `Store` object.

The only other situation in which this occurs is when a reactive function is being called on a `Store` object. For example, `mk.reactive(lambda x: x + 1)(x)` will be reactive, and return a `Store` object.

Finally, many other statements on `Store` objects will become reactive in the `magic` context. Read more about the `magic` context [XXXXXXXXX](XXXXX).

## How do I update a `Store`?

A `Store` can be updated by either the Python code and/or frontend.

- _Frontend_: If the store is used by a component on the frontend, changes on the frontend will automatically be reflected in the `Store` on the Python side.
- _Python_: The `Store` has a `.set` method that can be used to update its value. This `.set` method should only be used in functions decorated with `@mk.endpoint` if you want the updated `Store` to trigger re-execution of reactive functions.

```python
counter = mk.Store(0)
counter.set(1)
```

Remember, setting a store equal to a value (i.e. `store = 1`) will not update the value of the store. Use `store.set()`!

## Why does `Store(None) is None` return `False`?

Unfortunately, `is` is a special token in Python that cannot be intercepted by Meerkat, so this is an unavoidable limitation of the `Store` object. However, `Store(None) == None` will return `True`.

```python
Store(None) is None  # False
Store(None) == None  # True
```

## Do I need to do anything special inside reactive functions?

No. Write your reactive functions as you would any other Python function.

Reactive functions unwrap all `Store` arguments automatically, so treat those arguments as normal Python objects. They also automatically wrap the return value of the function in a `Store` object, so return any Python object you like.

```python
@mk.reactive()
def add(x: int, y: int):
    # Don't worry about `Store` objects here!
    # If x and y are passed in as Store objects,
    # they've already been unwrapped to their
    # wrapped values automatically.
    return x + y # this is just adding two `int`s

x = mk.Store(1)
y = mk.Store(2)
add(x, y) # returns a Store containing the value 3
```

Of course, you can still create `Store` objects manually inside reactive functions if you want to.

## Should I use the `magic` context?

It depends.

The `magic` context is a special context in which all methods and attribute access for `Store` objects become reactive i.e. `store.<blah>`-like statements (remember that `fn(store)` calls will not be reactive unless `fn` is explicitly decorated with `mk.reactive()` though!).

This means that you can use `Store` objects in a more natural way, without having to worry about whether or not the statement you're writing is reactive.

For example, instead of writing:

```python
df = mk.Store(pd.DataFrame({"a": [1, 2, 3]}))
# This doesn't work like a reactive statement
cols = df.columns
# You must wrap `df.columns` in a reactive function
# if you want the `cols` variable to be updated when `df` changes.
cols = mk.reactive(lambda df: df.columns)(df)
```

You can write:

```python
df = mk.Store(pd.DataFrame({"a": [1, 2, 3]}))
# No need to wrap `df.columns` in a reactive function
# if you're using the `magic` context.
with mk.magic():
    cols = df.columns
```

The `magic` context is not required at all, but it can make your code more succinct and easier to write. In general, we recommend using the `magic` context as little as possible, since it will make it much less explicit to a reader of your code whether or not a statement is reactive. Anything you can do in the `magic` context, you can also do without it.

## Are there any caveats to using the `magic` context?

There are some important edge cases when using the `magic` context that you should be aware of. You should see warnings when you encounter these edge cases, but it's still good to be aware of them.

One major edge case is when using `and`, `or`, `not`, `is`, and `in` operators. These operators are special tokens in Python, and cannot be intercepted by Meerkat. This means that the following statements will not be reactive, and should not be relied on.

```python
x = mk.Store(1)
y = mk.Store(2)
w = mk.Store([1, 2, 3])
with mk.magic():
    # None of these will be reactive
    # and may return the wrong value!
    z = x and y
    z = x is y
    z = x or y
    z = not x
    z = x in w
    # Use Meerkat's built-in overloads instead
    mk.cand(x, y)
    mk.cor(x, y)
    mk.cnot(x)
    # Or just wrap in a reactive lambda function!
    z = mk.reactive(lambda x, y: x and y)(x, y)
```

## What's a quick way to debug the value of `Store` objects?

The easiest way is to use the `mk.print` function. This is a reactive `print` function that will re-print the value of a `Store` object whenever it changes.

```python
x = mk.Store(1)
mk.print(x) # prints 1
# Will re-print the value of `x` whenever it changes
```

Underneath the hood, `mk.print` is just `mk.reactive(print)`, so you can also use it to print the value of any Python object.

You can follow this pattern to create more advanced debugging tools.

## Why do I get different `Store` objects when I index into a `Store` repeatedly?

This is a limitation of the `Store` object. When you index into a `Store` object (e.g. `store[0]`), you get a new `Store` object that wraps the value at that index. This means that if you index into a `Store` object multiple times, you will get different `Store` objects each time.

```python
x = mk.Store([1, 2, 3])
s1 = x[0]
s2 = x[0]
s1 is s2   # This is False

x = mk.Store({"a": 1, "b": 2})
s1 = x["a"]
s2 = x["a"]
s1 is s2   # This is False
```