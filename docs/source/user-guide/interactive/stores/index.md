(guide_store_getting_started)=

# Stores

<!-- In this section, we will discuss one of the core Markable objects in Meerkat: {class}`~meerkat.Store`. -->

An interactive application needs some way of creating variables that capture
the state of the application. This is important to

- keep track of application state that is going to change over time
- keep this state in sync between the frontend and Python code
- provide explicit ways to manipulate this state
- implement and debug the application in terms of this state

In Meerkat, wrapping a Python object in a {class}`~meerkat.Store` object provides a way to do this. These pages will explain:

- what a `Store` is
- how to work with `Store` objects
- how `Store` objects are used in interactive applications
- common pitfalls and how to avoid them

## What is a `Store`?

```{important}
A {class}`~meerkat.Store` behaves like the object it wraps, with extra functionality for reactivity.
```

A {class}`~meerkat.Store` is a special object provided by Meerkat that can be used to wrap arbitrary Python objects, such primitive types (e.g. int, str, list, dict), third-party objects (e.g. pandas.DataFrame, pathlib.Path), and even your custom objects.

For example, a `Store` can be used to wrap a string:

```python
x = mk.Store("Hello World!")
```

## How do we use a `Store`?

```{important}
`Store` objects are designed to be as transparent as possible, so that they behave like the object they wrap.
```

All attributes and methods of the wrapped object are exposed through the store.

For example, we can call `.lower()` on a `Store` wrapping a `str`.

```python
x = mk.Store('HELLO WORLD')
y = x.lower() # 'hello world'
type(y) # str
```

`Store` objects also behave exactly like the object they wrap when you access their attributes i.e. you use `.attribute` syntax. This is quite useful when wrapping complex objects, like {py:class}`pandas.DataFrame`.

```python
import pandas as pd
x = mk.Store(pd.DataFrame({'a': [1, 2, 3]}))
y = x.columns
type(y) # <class 'pandas.core.indexes.base.Index'>
```

### Using functions with `Stores`

`Store` objects behave (almost) exactly like the object they wrap when you use them as inputs to functions. For instance, the following code works as expected.

```python
x = mk.Store('HELLO WORLD')
y = len(x) # 11
type(y) # int
```

However, sometimes `Store` objects may behave unexpectedly when used with arbitrary functions. This mostly happens when functions are not duck-typed, but instead have behavior that expects specific types. It can also happen when a function is actually implemented in C.

Here's a simple example that fails from Python's `os` module.

```python
x = mk.Store('./relative/path/to/file')
y = os.path.abspath(x)
# TypeError: expected str, bytes or os.PathLike object, not Store
```

The workaround is simple. If you encounter an error of this kind, use the `.value` attribute to get the underlying object and pass that in to the function instead.

```python
x = mk.Store('./relative/path/to/file')
y = os.path.abspath(x.value)
type(y) # str
```

## Why do we need Stores?

As we saw earlier, stores are essential for building interactive interfaces - i.e. {ref}`reactivity <reactivity_getting_started>` and {ref}`endpoints <endpoints_getting_started>`.

```{important}
Stores are what make it possible to build reactive functions with arbitrary Python objects.
```

## Why do we need Stores?

The defacto way to mark an object is to wrap it in a Store. In this section, we will briefly cover the importance of Stores. You can read the full guide on Stores at [XXXXXXXXXXXX].

A Store is a special object provided by Meerkat that can be used to wrap arbitrary Python objects, such primitive types (e.g. int, str, list, dict), third-party objects (e.g. pandas.DataFrame, pathlib.Path), and even your custom objects. A major reason to use Store objects is that they make it possible for Meerkat to track changes to Python objects.

Let's look at an example to understand why they are so important to reactive functions.

@mk.reactive()
def add(a, b):
return a + b
Now, let's try something that might feel natural. Create two int objects, call add with them, and then change one of them.

x = 1
y = 2
z = add(x, y) # z is 3

x = 4 # z is still 3
You might think for a second that z should be updated to 6 because x changed and add is a reactive function. This is not the case.

This is because x is just an int. By changing x, we aren't changing the object that x points to (i.e. the int 1). Instead, we are just changing the variable x to point to a different object.

What we need here is to update the object that x points to. It's impossible to do this with a regular int. We can do this with a Store instead.

x = mk.Store(1)
y = mk.Store(2)
z = add(x, y) # z is Store(3), type(z) is Store

x.set(4, triggers=True)
print(z)

### z is now Store(6), type(z) is Store

By calling set() on x, we are changing the object that x points to. This is what allows z to be updated. Ignore the triggers=True argument for now, we discuss it in more detail in the section below.)

Store is a transparent wrapper around the object it wraps, so you can use that object as if the Store wasn't there.

```python
x = mk.Store(1)
y = mk.Store(2)
z = x + y # z is Store(3), type(z) is Store

message = mk.Store("hello")
message = message + " world"

# message is Store("hello world"), type(message) is Store
```

<!-- A very detailed breakdown of how Store objects behave is provided at XXXXXXX. We highly recommend reading that guide. -->

The takeaways are:

- A Store can wrap arbitrary Python objects.
- A Store will behave like the object it wraps.
- A Store is necessary to track changes when passing an object to a reactive function.
