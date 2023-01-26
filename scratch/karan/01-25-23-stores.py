import meerkat as mk


@mk.gui.react()
def foo(a: int, b: str):
    print("foo")
    print(a, b)


a = mk.gui.Store(1)
b = mk.gui.Store("hello")

c = foo(a=a, b=b)

@mk.gui.endpoint
def temp_setter(a: mk.gui.Store[int]):
    a.set(5)
    print(a)

print(a.inode.children)
temp_setter(a)
print(a.inode.children)

print("Info")
print(type(a), a.__wrapped__, getattr(a, "__eq__"))

with mk.gui.react():
    e = (a == 3)

temp_setter(a)

breakpoint()