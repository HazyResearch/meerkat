(reactivity_concepts_markables)=

## Markables

So far, we learned a simple definition of reactive functions in Meerkat. This definition was almost, but not quite, complete.

Rather than a reactive function being rerun whenever any of its inputs are updated, it is actually rerun whenever any of its **marked** inputs are updated. This is a subtle but important distinction.

**What objects can be marked?** By default, standard objects cannot be marked. However, they can be wrapped in a {class}`~meerkat.Store`. `Store` objects are _markable_, which means we can mark them for use with reactive functions. Python primitives, third-party objects, and custom objects can be wrapped in `Store` objects to make them markable. Other objects in Meerkat like `DataFrame` and `Column` are also markable. All of these objects provide a `.mark()` and `.unmark()` method to control whether they are marked or not.

This means that for a function to react to changes in its inputs, its inputs must be marked. Let's look at a few examples.

```{important}
Inputs into reactive functions must be **marked** for the function to react to changes in them.
```

```python
import meerkat as mk


@mk.reactive()
def add(a, b):
    return a + b


# c1 is updated if either a or b changes
a = mk.Store(1)
b = mk.Store(2)
c1 = add(a, b)

# c2 is updated only if a changes as b is unmarked
a = mk.Store(1)
b = mk.Store(2).unmark()
c2 = add(a, b)

# c3 is updated only if a changes as b is not markable
a = mk.Store(1)
b = 2
c3 = add(a, b)

# c4 will never be updated because neither a nor b are markable
a = 1
b = 2
c4 = add(a, b)
```

Most of the time, we won't need to worry about marking `Store` objects, since they are marked by default.

However, other Meerkat objects like `DataFrame` and `Column` are unmarked by default. If we want reactive functions to react to changes in them, we must call `.mark()` on them prior to passing them to a reactive function.

```python
@mk.reactive()
def df_sum(df):
    return df["a"].sum()


df = mk.DataFrame({"a": [1, 2, 3]})

# c1 is not updated if df changes because DataFrames are unmarked by default
c1 = df_sum(df)

# c2 is updated if df changes as df is now marked
df.mark()
c2 = df_sum(df)
```

Takeaways:

- Any inputs can be passed to reactive functions, but only marked inputs will be used to determine if the function should be rerun.
- Wrap a Python primitive or custom object in `Store` to make it markable.
- Use the `.mark()` and `.unmark()` methods to control whether a markable object is marked or not.
- `Store` objects are marked by default, whereas other Meerkat objects are unmarked by default.

---

---

---

<!-- #FIXME blend these two together -->

<!-- #FIXME make sure we did talk about the importance earlier -->

Earlier in this section, we mentioned the importance of marking inputs to reactive functions. Let's break down the principle behind marked objects and how unmarking them can actually be useful at times.

To recap, {class}`~meerkat.Store`, {class}`~meerkat.DataFrame`, and {class}`~meerkat.Column` objects are all markable objects. In order for a reactive function to rerun when an input changes, the input must be marked. We can toggle the mark on an object by calling `.mark()` or `.unmark()`.

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
@mk.reactive()
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
@mk.reactive()
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
