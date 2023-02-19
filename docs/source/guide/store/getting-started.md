# Stores

In this section, we will discuss one of the core Markable objects in Meerkat: {py:class}`Store <meerkat.Store>`.

## What is a `Store`?

```{important}
A {py:class}`Store <meerkat.Store>` behaves like the object it wraps, with extra functionality for reactivity.
```

A {py:class}`Store <meerkat.Store>` is a special object provided by Meerkat that can be used to wrap arbitrary Python objects, such primitive types (e.g. int, str, list, dict), third-party objects (e.g. pandas.DataFrame, pathlib.Path), and even your custom objects.

**How do we interact with a `Store`?**

All attributes and methods of the wrapped object are exposed through the store.
For example, if we wrap a `dict` in a `Store`, we can access the keys and values of the `dict` through the `Store`:

```python
store = mk.Store({"a": 1, "b": 2})
print(store.keys()) # ["a", "b"]
print(store.values()) # [1, 2]
```

You shouldn't have to worry (too much) about being able to use `Stores`, because all `Stores` behave just like their wrapped objects.


## Why do we need Stores?
As we saw earlier, stores are essential for building interactive interfaces - i.e. {ref}`reactivity <reactivity_getting_started>` and {ref}`endpoints <endpoints_getting_started>`.

### [Recap] Reactivity

### [Recap] Endpoints


## Why do we need Stores?

The defacto way to mark an object is to wrap it in a Store. In this section, we will briefly cover the importance of Stores. You can read the full guide on Stores at [XXXXXXXXXXXX].

A Store is a special object provided by Meerkat that can be used to wrap arbitrary Python objects, such primitive types (e.g. int, str, list, dict), third-party objects (e.g. pandas.DataFrame, pathlib.Path), and even your custom objects. A major reason to use Store objects is that they make it possible for Meerkat to track changes to Python objects.

Let’s look at an example to understand why they are so important to reactive functions.

@mk.reactive()
def add(a, b):
    return a + b
Now, let’s try something that might feel natural. Create two int objects, call add with them, and then change one of them.

x = 1
y = 2
z = add(x, y) # z is 3

x = 4 # z is still 3
You might think for a second that z should be updated to 6 because x changed and add is a reactive function. This is not the case.

This is because x is just an int. By changing x, we aren’t changing the object that x points to (i.e. the int 1). Instead, we are just changing the variable x to point to a different object.

What we need here is to update the object that x points to. It’s impossible to do this with a regular int. We can do this with a Store instead.

x = mk.Store(1)
y = mk.Store(2)
z = add(x, y) # z is Store(3), type(z) is Store

x.set(4, triggers=True)
print(z)
# z is now Store(6), type(z) is Store
By calling set() on x, we are changing the object that x points to. This is what allows z to be updated. Ignore the triggers=True argument for now, we discuss it in more detail in the section below.)

Store is a transparent wrapper around the object it wraps, so you can use that object as if the Store wasn’t there.

x = mk.Store(1)
y = mk.Store(2)
z = x + y # z is Store(3), type(z) is Store

message = mk.Store("hello")
message = message + " world" 
# message is Store("hello world"), type(message) is Store
A very detailed breakdown of how Store objects behave is provided at XXXXXXX. We highly recommend reading that guide.

The takeaways are:

A Store can wrap arbitrary Python objects.

A Store will behave like the object it wraps.

A Store is necessary to track changes when passing an object to a reactive function.