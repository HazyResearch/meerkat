import meerkat as mk

@mk.gui.endpoint
def foo():
    return

class Test(mk.gui.AutoComponent):

    a: list
    b: int
    c: int
    d: str = "blah"
    e: mk.gui.Endpoint

s = mk.gui.Store([1, mk.gui.Store(2), 3])
ss = mk.gui.Store(4)
sss = mk.gui.Store("hello")
test = Test(a=s, b=2, c=ss, d="hll", e=foo)
# test = Test(b=2, c=ss)
print(test)
print(s.id, ss.id, sss.id)
print(test.a.id, test.b.id, test.c.id, test.d.id)
print(test.a[1].id, s[1].id)
print(test.__fields__)
# print(test)
# print(test.b.id, test.c.id)

test1 = Test(a=s, b=2, c=ss, e=foo)
print(test1.d.id)
test2 = Test(a=s, b=2, c=ss, e=foo)
print(test2.d.id)