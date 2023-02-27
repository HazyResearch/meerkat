# Inbuilt Store Operators

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

| name                         | symbol       | example  |
| :--------------------------- | :----------- | :------- |
| Addition                     | +            | x + y    |
| Subtraction                  | -            | x - y    |
| Multiplication               | \*           | x \* y   |
| Division                     | /            | x / y    |
| Floor Division               | //           | x // y   |
| Modulo                       | %            | x % y    |
| Exponentiation               | \*\*         | x \*\* y |
| Add & Assign                 | +=           | x+=1     |
| Subtract & Assign            | -=           | x-=1     |
| Multiply & Assign            | \*=          | x\*=1    |
| Divide & Assign              | /=           | x/=1     |
| Floor Divide & Assign        | //=          | x//=1    |
| Modulo & Assign              | %=           | x%=1     |
| Exponentiate & Assign        | \*\*=        | x\*\*=1  |
| Power & Assign               | \*\*=        | x\*\*=1  |
| Bitwise Left Shift & Assign  | <<=          | x<<=1    |
| Bitwise Right Shift & Assign | >>=          | x>>=1    |
| Bitwise AND & Assign         | &=           | x&=1     |
| Bitwise XOR & Assign         | ^=           | x^=1     |
| Bitwise OR & Assign          | \|=          | x\|=1    |
| Bitwise Left Shift           | <<           | x << y   |
| Bitwise Right Shift          | >>           | x >> y   |
| Bitwise AND                  | &            | x & y    |
| Bitwise XOR                  | ^            | x ^ y    |
| Bitwise OR                   | \|           | x \| y   |
| Bitwise Inversion            | ~            | ~x       |
| Less Than                    | <            | x < y    |
| Less Than or Equal           | <=           | x <= y   |
| Equal                        | ==           | x == y   |
| Not Equal                    | !=           | x != y   |
| Greater Than                 | >            | x > y    |
| Greater Than or Equal        | >=           | x >= y   |
| Get Item                     | [key]        | x[0]     |
| Get Slice                    | [start:stop] | x[0:10]  |

<!---autogen-end: mk-store-reactive-operators-->

- Store operations are reactive
- Don't do in-place operations without calling .set in an endpoint
- interface with magic
