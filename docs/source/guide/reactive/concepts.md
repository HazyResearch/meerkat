(reactivity_concepts)=

# Concepts

In this guide, we'll cover high level concepts related to reactivity. If you haven't already, please read {ref}`Getting Started <reactivity_getting_started>`, which will provide an introduction to reactivity.

(reactivity_concepts_stores)=

## Recap: Stores

Recall, a core principle of reactivity is that Meerkat only tracks **marked** inputs into functions. If an input is not marked, Meerkat will not track it and will not rerun the function if it changes.

The defacto way to mark an object is to wrap it in a {py:class}`Store <meerkat.Store>`. In this section, we will briefly cover the importance of the `Store` object. You can read the full `Store` guide {ref}`here <guide_store_getting_started>`.

A `Store` is a special object provided by Meerkat that can be used to wrap arbitrary Python objects such as primitive types (e.g., `int`, `str`, `list`, `dict`, etc.), third-party objects (e.g., {py:class}`pandas.DataFrame`, `pathlib.Path`), and even our own custom objects. `Store` objects make it possible for Meerkat to track changes to Python objects.

Let's look at an example to understand why they are so important to reactive functions.

```python
@mk.reactive()
def add(a, b):
    return a + b
```

Now, let's try something that might feel natural. We'll create two `int` objects, call `add` with them, and then change one of them.

```python
x = 1
y = 2
z = add(x, y)  # z is 3

x = 4  # z is still 3
```

You might think for a second that `z` should be updated to `6` because `x` changed and `add` is a reactive function. **This is not the case.**

This is because `x` is just an `int`. By changing `x`, we aren't changing the object that `x` points to (i.e., the `int` `1`). Instead, we are just changing the variable `x` to point to a different object.

_We need to update the object that `x` points to._ It's impossible to do this with a regular `int`. We can do this with a `Store` instead.

```python
x = mk.Store(1)
y = mk.Store(2)
z = add(x, y)  # z is Store(3), type(z) is Store

x.set(4, triggers=True)
# z is now Store(6), type(z) is Store
```

By calling `set()` on `x`, we are changing the object that `x` points to. This is what allows `z` to be updated.

```{note}
`triggers=True` is an optional argument to `Store.set()` to trigger the reactive chain. It should never be called explicitly in the code but can be helpful for debugging purposes.
<!-- #FIXME - why are we calling it explicitly then? -->
```

`Store` is a _transparent wrapper_ around the object it wraps, so we can use that object as if the `Store` wasn't there.

```python
x = mk.Store(1)
y = mk.Store(2)
z = x + y  # z is Store(3), type(z) is Store

message = mk.Store("hello")
message = message + " world"  # message is Store("hello world"), type(message) is Store
```

We highly recommend reading the detailed breakdown of how `Store` objects behave, which can be found here XXXXXXX (#FIXME).

The takeaways are:

- A `Store` can wrap arbitrary Python objects.
- A `Store` will behave like the object it wraps.
- A `Store` is necessary to track changes when passing an object to a reactive function.

(reactivity_concepts_reactive_functions)=

## Demystifying Reactive Functions

Now that we understand the importance of `Store` objects to reactivity, let's look at how reactive functions work. Consider the same function `add`.

**What happens when you wrap `add` with `mk.reactive()`?**

```python
@mk.reactive()
def add(a, b):
    return a + b
```

Meerkat tracks the function `add` and reruns it if any of the inputs to it (i.e., `a` or `b`) change. Here, we are assuming that both `a` and `b` are {ref}`marked <reactivity_concepts_markables>`. Internally, Meerkat puts the `add` function into a computation graph that tracks dependencies between reactive functions.

**What happens when you call `add`?**

```python
x = mk.Store(1)
y = mk.Store(2)
z = add(x, y)  # z is Store(3), type(z) is Store
```

Meerkat does the following steps:

1. **Unwrap input stores.** Any inputs to `add` that are `Store` objects are unwrapped before passing them into the body of `add`.
2. **Turn off reactivity.** Meerkat turns off reactivity before executing the function. This only applies for the duration of the function call.
3. **Execute the function.** The function is executed as if it were a normal Python function.
4. **Wrap and mark the outputs**: Then, the output of `add` is wrapped in a `Store` object and returned. If the output is a markable object (e.g., `mk.DataFrame <meerkat.DataFrame>`), the object is marked. This lets us use the output of `add` as input to other reactive functions.
5. **Turn on reactivity:** Meerkat turns reactivity back on once the function has returned.

The main consequence of these steps is that **any reactive function can be written as a regular Python function**. We don't have to worry about unwrapping `Store` inputs or managing reactivity manually. Thus, we can convert any existing Python function into a reactive function simply by wrapping it with `mk.reactive()`.

**What happens when you change any of the inputs to `add`?**

```python
x = mk.Store(1)
y = mk.Store(2)
z = add(x, y)  # z is Store(3), type(z) is Store

x.set(4, triggers=True)
# z is now Store(6), type(z) is Store
```

What does Meerkat do when `x` is changed? Let's walk through the steps:

- Meerkat detects that `x` changed because `x` is a `Store`, and `.set()` was called on it.
- Meerkat looks for all reactive functions that were called with `x` as an argument (i.e. `add`).
- Meerkat then reruns those functions (i.e., `add`) and any functions that depend on their outputs.
- Finally, Meerkat updates the outputs of any reactive functions that it reruns (i.e. `z`).

(reactivity_concepts_markables)=

## Markables and Reactivity

Earlier in this section, we mentioned the importance of marking inputs to reactive functions. Let's break down the principle behind marked objects and how unmarking them can actually be useful at times.

To recap, {py:class}`Store <meerkat.Store>`, {py:class}`mk.DataFrame <meerkat.DataFrame>`, and {py:class}`Column <meerkat.DataFrame>` objects are all markable objects. In order for a reactive function to rerun when an input changes, the input must be marked. We can toggle the mark on an object by calling `.mark()` or `.unmark()`.

```python
# Stores are marked by default.
x = mk.Store(1)

df = mk.DataFrame({"a": [1, 2, 3]}).mark()
col = mk.ScalarColumn([1, 2, 3]).mark()
```

### Understanding the `marked` state

```{important}
The marked state of an input controls whether or not the function will be rerun when the value of the input changes.
```

Marking an input indicates to a reactive function that changes to the input should trigger the function to rerun.
If an input is unmarked or unable to be marked, the function will not be triggered to rerun when the input changes.

Note that it is at the time the function is called, that an input's marked state determines if that function will retrigger.

**What happens when we unmark an object after passing it to a reactive function?**

Nothing!
If the input was marked at the time the reactive function was run, the function will react to changes in that input, even if the input is subsequently unmarked.
The marked state is only read when the function is called.

Consider the case where we unmark `a` after passing it to `add`.

```python
@mk.gui.reactive()
def add(a, b):
    return a + b

a = mk.Store(1)
b = mk.Store(2)
c = add(a, b) # c is Store(3), type(c) is Store

a.unmark()
d = add(a, b)
# d only updates when b changes, because a was unmarked when passed to the function that returned d.
# However, c will still update when a changes. This is because a was marked when passed to the function that returned c.
```

### Unmarking inputs to reactive functions

It seems natural to only want to pass marked inputs to reactive functions.
This way we can be sure that the function will rerun when any of its input changes.

```{important}
Unmarked inputs to reactive functions will not trigger the function to rerun.
However, if the function is rerun, the newest value of the unmarked input will be used.
```

However, in some cases, we may want to be selective about which inputs should trigger the function.
Perhaps, we only want to trigger the function when a certain input changes.

To achieve this, we can simply unmark the inputs that we don't want to trigger the function.
For example, say we do not want `add` to rerun when `a` changes, but we do want it to rerun when `b` changes.

```python
@mk.gui.reactive()
def add(a, b):
    return a + b

a = mk.Store(1).unmark()
b = mk.Store(2)
c = add(a, b)

a.set(4, triggers=True)
# c is still Store(3) because a was unmarked when passed to the function that returned c.
# Thus changes in a will not trigger the function to rerun.
```

### Non-markable inputs to reactive functions

We may not always pass markable objects to reactive functions.
For example, I may pass a list to a function.

If a object that is not markable is passed to a reactive function, the function will not rerun when the object changes.
However, if the function is ever retriggered (e.g. by changing another marked input into the function), the newest value of the unmarked input will be used.
**This only works when the object is modified in-place** - i.e. the input needs to be mutable.
To understand why this happens, see the discussion on {ref}`pass-by-assignment <reactive_concepts_pass_by_reference>`.

A great example of this is a `list`. Below, we define a function that takes a list and a value, and returns the sum of the list and the value. Note that a list, when it is not wrapped in a `Store`, is not a markable object.

```python
def sum_list_with_value(my_list: list, x: int):
    return sum(my_list) + value

my_list = [1, 2, 3]
x = mk.Store(4)
out = sum_list_with_value(my_list, x)
print("out", type(out), out)
# out is Store(10)

my_list.append(4)  # my_list = [1, 2, 3, 4]
print("out", type(out), out)
# out is still Store(10) because the function was not rerun when my_list changed.

x.set(5, triggers=True)
# The value of x changed with `triggers=True`, so the function was rerun.
print("out", type(out), out)
# out is now Store(15) - sum([1, 2, 3, 4]) + 5
```

(reactive_concepts_pass_by_reference)=

### [Aside] Pass by Reference vs Pass by Value

Recall, Python passes arguments by assignment.
If you are not familiar with passing by assignment, we can very loosely boil it down to this:

- immutable objects (e.g. primitives - `int`, `float`, `str`, `bool`, etc.) are passed by value
- mutable objects (e.g. `list`) are passed by reference

_NOTE_: This is not a perfect analogy, but it is sufficient for our purposes.

Reactive functions are just like regular Python functions in this way.
Immutable inputs into a reactive function can never change. If the reactive function is rerun, the input will be the same as it was the first time.
In contrast, mutable inputs can change. If the input was modified in-place and the reactive function is rerun, the input will be different than it was the first time.

This does not mean that if any mutable inputs are changed, the reactive function will rerun.
The mutable input must both be `marked` and be changed in a special way `.set()` to trigger
the reactive function to rerun. We will see why this is important in the next section.

Any markable object in Meerkat is a mutable object.
This means there are operations that can be performed on that object in-place.
For example, we can set the value of a `Store` using `.set()`.

```python
a = mk.Store(1)
print(id(a))

a.set(2)  # modifies the store `a` in-place
print(id(a))
```

## Reactivity as a Graph

We can think of reactivity as a graph of inputs, functions, and outputs.
In fact, Meerkat maintains a graph of reactive functions and their inputs and outputs to determine what functions should be rerun.

**What are the nodes?**

Reactive functions and their inputs and outputs are all nodes in the graph.

When a reactive function is called with inputs, a node is created for the function call and its inputs.
The node for the function call is referred to as an `Operation`.
The operation corresponds to this particular function call. If the function is called again, a new operation is created.

**What are edges in this graph?**
Edges indicate what variables are inputs to a function, and what variables are outputs of a function.
These edges are directed - i.e. input -> operation -> output.

There are two kinds of edges in the graph:

1. **Trigger Edges**: These edges indicate which variables should trigger an operation to rerun.
2. **Connector Edges**: These edges simply indicate that variables are inputs/outputs (i.e. connected) to an operation. Connector edges between an input and an operation indicate that changes in the input will not trigger the operation to rerun.

Trigger edges are used to describe relationships between inputs and the operation. In other words,
an input _triggers_ an operation to rerun if the input points to the operation with a trigger edge.

This distinction is important because it allows us to be selective about which inputs should trigger a function to rerun.
For example, `c = add(a,b)` only reruns when `b` changes.

```python
@mk.reactive()
def add(a, b):
    return a + b

a = mk.Store(1).unmark()
b = mk.Store(2)
c = add(a, b)
```

The graph looks like this:

```{figure} ../../../assets/guide/graph/add.png

```

TODOS:

- What happens when reactive function doesnt have arguments
- Using the same reactive function multiple times
