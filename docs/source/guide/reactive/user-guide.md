
# User Guide

## Chaining Reactive Functions
One of the reasons reactive functions are so powerful is that they can be chained together.
This allows you to break up your code into nicely abstracted functions, and then compose them together to create complex workflows.

Let's look at an example with two reactive functions, `add` and `multiply`.
```python
@mk.reactive()
def add(a, b):
    return a + b

@mk.reactive()
def multiply(a, b):
    return a * b

x = mk.Store(1)
y = mk.Store(2)

u = add(x, y) # u is Store(3), type(u) is Store
v = add(y, 7) # v is Store(9), type(v) is Store
w = multiply(u, 2) # w is Store(6), type(w) is Store

x.set(4, triggers=True)
# u is now Store(6), type(u) is Store
# v is not changed
# w is now Store(12), type(w) is Store
```
Let's break down this example:
- `add` and `multiply` are both reactive functions.
- `u` depends on `x` and `y`, so it will be updated if either of them are updated.
- v depends only on `y`, so it will only be updated if `y` updates.
- w depends on `u`, so it will be updated if `u` is updated.

When we update `x`, only `u` and `w` are updated, by running `add` and `multiply` again respectively. If instead we updated `y`, all three outputs `u`, `v` and `w` would be updated, with `add` running twice and `multiply` running once.

In practice, Meerkat provides reactive functions for many common operations on `Store` objects. So we could have written the example above as:
```python
x = mk.Store(1)
y = mk.Store(2)

u = x + y # u is Store(3), type(u) is Store
v = y + 7 # v is Store(9), type(v) is Store
w = u * 2 # w is Store(6), type(w) is Store
```
You can find a full list of the reactive functions that Meerkat provides at XXXXXXX, and more information about reactivity with `Store` objects at XXXXXXX.

One place where chaining can be helpful is when you want to create a function that takes a DataFrame as input, and returns another DataFrame as output (a "view" of the original DataFrame). Let's look at an example.

```python
@mk.reactive()
def filter(df, column, value):
    return df[df[column] == value]

@mk.reactive()
def sort(df, column):
    return df.sort(column)

@mk.reactive()
def select(df, columns):
    return df[columns]

df = mk.DataFrame({
    "a": [random.randint(1, 10) for _ in range(100)], 
    "b": [random.randint(1, 10) for _ in range(100)],
})
filtered_df = filter(df, "a", 2)
sorted_df = sort(filtered_df, "b")
selected_df = select(sorted_df, ["b"])
```
Here, we filter, sort and select columns from a `mk.DataFrame`, chaining these reactive functions together to achieve this effect. Again, in practice, Meerkat provides reactive functions for many common operations on `mk.DataFrame`, so we could have written the example above as:
```python
filtered_df = df[df["a"] == 2]
sorted_df = filtered_df.sort("b")
selected_df = sorted_df[["b"]]
```

> This pattern works with `mk.DataFrame` and not `pandas.DataFrame`. If you're using a `pd.DataFrame`, we recommend converting to a `mk.DataFrame` using `mk.DataFrame.from_pandas(df)` when using Meerkat's interactive features.
> 
> Alternately, you can also use the `magic` context manager if you want to "turn on" the same functionality for `pd.DataFrame` objects. We recommend reading the guide at XXXXXXX to learn more about this.

## Guidelines for Writing Reactive Functions
Let's go over a few guidelines for writing reactive functions.

**Write reactive functions like normal functions**
Write reactive functions like any other Python function. There's no need to think of inputs to the function in any special way, since `Store` objects will be unwrapped by Meerkat to their underlying values automatically.

The only case in which it doesn't make sense to wrap a normal function with the `@mk.reactive` decorator is for functions that take no arguments, since they can never be re-run.


An important rule of thumb when using reactive functions is: **don't edit objects inside them without thinking very carefully about the consequences.**

This isn't a strict rule, but it's a good guideline to follow. Let's look at an example that is particularly bad practice:
```python
@mk.reactive()
def foo(df: mk.DataFrame):
    # .... do some stuff to df
    df["a"] += 1
    return df

df = mk.DataFrame({"a": [1, 2, 3]})
df_2 = foo(df)
```
Here, `foo` quietly updates the original `df` object, and then returns it. This is bad practice for a couple of reasons:
- The output of `foo` is the same object as the input to it, which leads to a cyclical dependency for this reactive function. This means that if `df` is updated, `foo` will be re-run, and then `df` will be updated again, and so on.
- Meerkat disregards when `df` is updated in-place inside a reactive function, so it won't update any outputs that depend on `df`. If it did, it would cause an infinite loop (since `foo` must have been called because `df` was updated, and so on).

The way to avoid this is to just not edit objects in-place inside reactive functions. Generally, this will also mean you won't return the same object as the input into the function. Instead, create a new object and return that. For example:
```python
@mk.reactive()
def foo(df: mk.DataFrame):
    # .... do some stuff to df
    df_2 = df.copy()
    df_2["a"] += 1
    return df_2
```
This is a much better way to write `foo`, because it doesn't edit the original `df` object, and it doesn't return the same object as the input into it.

The appropriate place to edit objects in-place in response to user input is inside a `@mk.endpoint()` function, which you can read more about in the guide at XXXXXXX.


**On type hints**
Generally, we recommend that you type-hint the inputs to a reactive function without using `Store` objects, since they will be automatically unwrapped inside the function body.

```python
# Do this.
@mk.reactive()
def add(a: int, b: int):
    return a + b

# Don't do this.
@mk.reactive()
def add(a: mk.Store, b: mk.Store):
    return a + b
```

For return values, type-hinting the return value as a `Store` object is better, since the return value will be wrapped into a `Store` automatically.

```python
@mk.reactive()
def add(a: int, b: int) -> mk.Store[int]:
    return a + b
```

**Use reactive functions over shortcuts**

Another guideline we recommend is to actually write reactive functions rather than using shortcuts such as direct operations on `Store` objects, or overusing the `magic` context. If you do use shortcuts, we recommend you put them inside a `magic` context for readability.

As an example,
```python
a = mk.Store(1)

# Method 1: best and should be preferred, overkill for this example
@mk.reactive()
def add_five(x: int):
    return x + 5
b = add_five(a)

# Method 2: inlined reactive function is quite readable
b = mk.reactive(lambda x: x + 5)(a)

# Method 3: Store shortcut on + is convenient, but not readable
b = a + 5

# Method 4: same shortcut, but magic context makes it more readable
with magic():
    b = a + 5
```

**How to think about return values of reactive functions.**

When you write a reactive function, the return value will automatically be wrapped in a **single** `Store` object that is created by the function, **regardless of what was returned**. This is different from how you might think about return values in a normal Python function, where the return value is just a reference to an existing object.

Let's go over the consequences of this over a few different return value types.

If you return a single object of any type, it will be wrapped in a `Store` object. The only exception is other Meerkat objects like `mk.DataFrame` and `mk.Column`, which never to be wrapped by `Store`. 

To fix an example, we'll take the following snippet of code and think about what `a` will be for different return values of `foo`.
```python
@mk.reactive()
def foo(...) -> ...:
    return ...

a = foo()
```
To be explicit, here are some examples:

| Return Value | `a` |
| --- | --- |
| `1` | `Store(1)` |
| `"hello"` | `Store("hello")` |
| `(1, 2)` | `Store((1, 2))` |
| `[1, 2]` | `Store([1, 2])` |
| `{"a": 1}` | `Store({"a": 1})` |
| `mk.DataFrame({"a": [1, 2, 3]})` | `mk.DataFrame({"a": [1, 2, 3]})` |

In all cases, `a` will be a `Store`, except if the return value is a Meerkat object.

**Notes on nested return values.**

There are cases where you might wonder how to return nested `Store` objects. For example, you might wonder how to return a tuple of `Store` objects, rather than only a single `Store` object containing a tuple e.g. maybe you want to return a pair of `int` values, where each value is wrapped in a `Store` object.

This is generally not something you will need to ever do explicitly. Let's look at a simple example where we chain two reactive functions to understand why.
```python
@mk.reactive()
def foo():
    return 1, 2 # Return a tuple of ints

@mk.reactive()
def bar(a):
    return a + 1

# Chain: foo()[0] -> bar()
x = foo() # Store((1, 2))
y = bar(x[0])
```
What happens when we pass `x[0]` to `bar`? Indexing into a `Store` is actually reactive, so `x[0]` will *itself* behave like a reactive function that takes `0` as input and returns `x[0]` as output _wrapped in a `Store`_. Passing this `Store` on to `bar` means that `y` will be re-run if `x` changes, which is exactly what we want.

If indexing into a `Store` with `x[i]` was not reactive, it would actually _break the chain of reactivity_, and `bar` would not be re-run if `x` changed.

The takeaway is that Meerkat obviates the need to explicitly wrap nested return values in `Store` objects, since indexing into a `Store` is itself reactive.

However, keep in mind that when indexing into a `Store` to access nested values, a new `Store` object is created each time, even if you index into the same location. This means that if you want to access the same nested value multiple times, you should store it in a variable first.

```python
x = Store((1, 2))
a = x[0]
b = x[0]
# `a` and `b` are different `Store` objects.
# This can lead to subtle bugs e.g. setting `a` would not trigger
# functions that depend on `b`.
```
The solution here is to just access the nested value once to define `a`, and then reuse the `a` variable.


Finally, if you do want to explicitly control how return values of a reactive function are wrapped by `Store`, you have the option to wrap these values in a `Store` inside the function and return it yourself.
```python
@mk.reactive()
def foo():
    return mk.Store(1), mk.Store(2)

a = foo()
# `a` will be Store((Store(1), Store(2)))
```
In this case, the return value is a tuple of `Store` objects, and so `a` will be a `Store` object containing a tuple of `Store` objects.




