# Limitations

{py:class}`mk.magic <meerkat.magic>` can be quite powerful and very natural to use. However, there are some limitations to be aware of.

## Unsupported Operators
Using `stores` with `and`, `or`, `not`, `is`, and `in` operators will not be reactive. These operators are special tokens in Python, and cannot be intercepted by Meerkat. This means that the following statements will not be reactive, and should not be relied on.

```python
x = mk.Store(1)
y = mk.Store(2)
w = mk.Store([1, 2, 3])

with mk.magic():
    # None of these will be reactive
    # and may return the wrong value!
    z = x and y
    z = x or y
    z = not x
    z = x is y
    z = x in w
```

Meerkat provides reactive alternatives for some of these operators, which can be found in the {ref}`reactivity inbuilts <reactivity_inbuilts>`.

```python
x = mk.Store(1)
y = mk.Store(2)
w = mk.Store([1, 2, 3])

z = mk.cand(x, y)
z = mk.cor(x, y)
z = mk.cnot(x, y)
```

Alternatively, you can wrap the expression in a reactive lambda function.

```python
x = mk.Store(1)
y = mk.Store(2)
w = mk.Store([1, 2, 3])

z = mk.reactive(lambda x, y: x and y)(x, y)
```


## Calling in-place methods

```{important}
In-place methods should almost always be called outside of the `magic` context.
```

In Python, mutable objects often have methods that modify the object in-place. For example, `list.append` modifies the list in-place, and does not return a new list.

As we mentioned in the {ref}`reactivity <reactivity_getting_started>` guide, using in-place methods should almost always be avoided in reactive functions. 

<!-- This is because the object will be modified in-place when the function is re-run, the object will be modified in-place, and the result will be different. -->

`magic` does not distinguish between in-place and out-of-place methods. This means that if you call an in-place method on a `store` within the `magic` context, you will have made an in-place function reactive.

Let's look at an example with `list.append`. Calling `.append` on a list will append the item to the list in-place, and return `None`. Now, if we call `list.append` on a `store` within the `magic` context, the `store` will be modified in-place and the `.append` call will be added to the graph.

```python
x = mk.Store([1, 2, 3])

with mk.magic():
    x.append(4)
```

Because `.append` is on the graph, any time the store is updated, `4` will be appended to the list. This is almost certainly not what you want.

```python
x = mk.Store([1, 2, 3])

with mk.magic():
    x.append(4)

# Update the store with a new list.
x.set([7, 8, 9], triggers=True)

# Because we have set the store and `.append(4)` was added
# to the graph, `.append(4)` will be called again.
# x = [7, 8, 9, 4]
print("x", x)
```

Examples of common in-place methods include:

<!---autogen-start: common-inplace-methods-->
- {py:meth}`dict.clear`
- {py:meth}`dict.pop`
- {py:meth}`dict.popitem`
- {py:meth}`dict.setdefault`
- {py:meth}`dict.update`
- {py:meth}`list.append`
- {py:meth}`list.clear`
- {py:meth}`list.extend`
- {py:meth}`list.insert`
- {py:meth}`list.pop`
- {py:meth}`list.remove`
- {py:meth}`list.reverse`
- {py:meth}`list.sort`
- {py:meth}`set.add`
- {py:meth}`set.clear`
- {py:meth}`set.discard`
- {py:meth}`set.pop`
- {py:meth}`set.remove`
- {py:meth}`set.update`
<!---autogen-end: common-inplace-methods-->

