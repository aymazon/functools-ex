# functools-ex

## Instruction
It depends on fn/toolz/pyrsistent, and provide faster implementation, especially on PyPy.

It is probably the fastest open-resource functional base functions on PyPy.

You can use cython to build these functions on CPython, it will be faster than others,
but I do not think that is necessary, before your doing, read about performance,
[PyPy](https://www.pypy.org/performance.html), [pyrsistent](https://github.com/tobgu/pyrsistent#performance).


Faster functions:
    ("flip", "F", "FF(thread_last)", "X(_)", "R(juxt)", "fold")

New functions:
    ("op_set", "op_get", "tco_yield")

## Exemples

1. flip, faster than fn and toolz.

```python
>>> from operator import sub
>>> flip(sub)(3, 2)
-1
```

1. F, like [fn.F](https://github.com/kachayev/fn.py#high-level-operations-with-functions).

```python
>>> from functoolsex import F
>>> from functools import partial
>>> from operator import add
>>> F(add, 1)(2) == partial(add, 1)(2)
True
>>> from operator import add, mul
>>> (F(add, 1) >> (mul, 3))(2)
9
>>> (F(add, 1) << (mul, 3))(2)
7
```

1. FF, like [toolz.thread_last](https://github.com/pytoolz/toolz/blob/ea3ba0d60a33b256c8b2a7be43aff926992ffcdb/toolz/functoolz.py#L78).

```python
>>> from functoolsex import FF
>>> from operator import add, mul
>>> inc = lambda x: x + 1
>>> FF(2, inc, (mul, 3))
9
>>> FF(2, (mul, 3), inc)
7
```

1. X, like [fn._](https://github.com/kachayev/fn.py#scala-style-lambdas-definition).
But not support (_ + _ and print its definition), because it is terribly slow.
You can find more examples in functoolsex/tests.py.

```python
from functoolsex import X
class A():
    x = 'a'

assert (X == 'A')('A')
assert (X.x.call('upper') == 'A')(A())
assert ((X.call('lower').upper)('a'))() == 'A'
assert (X[0][1] == 2)([(1, 2), (3, 4)])
```

1. R, like [toolz.juxt](https://github.com/pytoolz/toolz/blob/ea3ba0d60a33b256c8b2a7be43aff926992ffcdb/toolz/functoolz.py#L646).
Run functions with the same args, support tuple partial.

```python
>>> from functoolsex import R, X
>>> from operator import add
>>> from functools import partial
>>> from toolz import juxt
>>> R(X + 1, (add, 1))(2)
(3, 3)
>>> R(X + 1, (add, 1))(2) == juxt(X + 1, partial(add, 1))(2)
True
>>> R(X + 1, (add, 1))(2) == ((X + 1)(2), add(1, 2))
True
>>> def func(a, b):
...     def fa():
...         return a
...     def fb():
...         return b
...     return R(fa, fb)()
>>> func(1, 2)
(1, 2)
```

1. (op_set, op_get), provide filter ability, take the place of [fn.monad.Option](https://github.com/kachayev/fn.py#functional-style-for-error-handling).

```python
>>> from functoolsex import F, X
>>> from operator import add
>>> (F(op_set) >> (filter, X == 1) >> (map, F(add, 1)) >> (op_get, -1))(1)
2
>>> (F(op_set) >> (filter, X > 1) >> (map, F(add, 1)) >> (op_get, -1))(1)
-1
```

1. fold, like [toolz.sandbox.fold](https://github.com/pytoolz/toolz/blob/ea3ba0d60a33b256c8b2a7be43aff926992ffcdb/toolz/sandbox/parallel.py#L13), but as fast as reduce on PyPy.

```python
>>> from functoolsex import fold
>>> from operator import add
>>> fold(add, range(10))
45
>>> from multiprocessing import Pool, cpu_count
>>> pool_map, pool_size = Pool().imap, cpu_count()
>>> fold(add, range(1000), map=pool_map, pool_size=pool_size)
499500
>>> fold(add, range(1000), 1, map=pool_map, pool_size=pool_size)
499501
```

1. tco_yield, while is bad, and yield from is terribly slower.
More info [fn.recur.tco](https://github.com/kachayev/fn.py#trampolines-decorator).

```python
>>> from functoolsex import tco_yield
>>> def read_line(fp):
...     @tco_yield()
...     def go(i):
...         line = fp.readline()
...         if line:
...             return None, f"{line[:-1]} {i}", (i + 1, )
...     return go(0)

>>> with open('tmp.txt', 'r') as fp:
...     print(list(read_line(fp)))
```


# Ignore these
```bash
# edit tag in setup.py
git tag v0.0.1
rm dist/functoolsex-*
python setup.py sdist bdist_wheel
twine upload dist/*
```
