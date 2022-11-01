# functools-ex

## Instruction
It depends on fn/toolz/pyrsistent, and provides faster implementation, especially on PyPy.
Now for > Python 3.11, fn is merged.

It is probably the fastest open-resource functional base functions on PyPy.

You can use cython to build these functions on CPython, it will be faster than others,
but I do not think that is necessary, before your doing, read about performance,
[PyPy](https://www.pypy.org/performance.html), [pyrsistent](https://github.com/tobgu/pyrsistent#performance).


```
Faster functions:
    ("flip", "C", "F", "FF(thread_last)", "X(_)", "R(juxt)", "fold")

New functions:
    ("tco_yield", "op_filter", "op_map", "op_or_else", "op_or_call", "op_get_or", "op_get_or_call",
     "e_filter", "e_left", "e_right", "e_is_left", "e_is_right", "e_map", "e_or_else", "e_or_call",
     "e_get_or", "e_get_or_call", "e_get_or_raise", "e_get_left", "e_get_right",
     "R", "fold", "is_none", "is_not_none", "is_option_full", "is_option_empty", "uppack_args", "log")
PS:
    Log is very useful to debeg, and can be close with env PY__FUNCTOOLSEX_LOG_OFF.
```

## Examples

### flip, faster than fn and toolz.
```python
>>> from operator import sub
>>> flip(sub)(3, 2)
-1
```

### P, just alias to functools partial

### C, Curry a function.
```python
>>> from functoolsex import C
>>> sum_func = lambda x, y, z: x + y + z
>>> sum_C = C(sum_func, 3)
>>> sum_C(1, 2, 3)
6
>>> sum_C(1, 2)(3)
6
>>> sum_C(1)(2)(3)
6
>>> sum_C = C(sum_func, 3)
>>> sum_C(1)(2)(3)
6
```

### F, like [fn.F](https://github.com/kachayev/fn.py#high-level-operations-with-functions).
```python
>>> from functoolsex import F, P
>>> from functools import partial
>>> from operator import add
>>> F(add, 1)(2) == partial(add, 1)(2)
True
>>> from operator import add, mul
>>> (F(add, 1) >> P(mul, 3))(2)
9
>>> (F(add, 1) << P(mul, 3))(2)
7
```

### FF, like [toolz.thread_last](https://github.com/pytoolz/toolz/blob/ea3ba0d60a33b256c8b2a7be43aff926992ffcdb/toolz/functoolz.py#L78).
```python
>>> from functoolsex import FF, P
>>> from operator import add, mul
>>> inc = lambda x: x + 1
>>> FF(2, inc, P(mul, 3))
9
>>> FF(2, P(mul, 3), inc)
7
```

### X, like [fn.\_](https://github.com/kachayev/fn.py#scala-style-lambdas-definition). But not support (_ + _ and print its definition), because it is terribly slow. You can find more examples in functoolsex/tests.py.
```python
from functoolsex import X
class A():
    x = 'a'

assert (X == 'A')('A')
assert (X.x._('upper') == 'A')(A())
assert ((X._('lower').upper)('a'))() == 'A'
assert (X[0][1] == 2)([(1, 2), (3, 4)])
```

### R, like [toolz.juxt](https://github.com/pytoolz/toolz/blob/ea3ba0d60a33b256c8b2a7be43aff926992ffcdb/toolz/functoolz.py#L646). Run functions with the same args.
```python
>>> from functoolsex import R, X, P
>>> from operator import add
>>> from functools import partial
>>> from toolz import juxt
>>> R(X + 1, P(add, 1))(2)
(3, 3)
>>> R(X + 1, P(add, 1))(2) == juxt(X + 1, partial(add, 1))(2)
True
>>> R(X + 1, P(add, 1))(2) == ((X + 1)(2), add(1, 2))
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

### op, '>>' can format by editor, take the place of [fn.monad.Option](https://github.com/kachayev/fn.py#functional-style-for-error-handling).
```python
>>> from functoolsex import F, X, P
>>> from operator import add
>>> (F(op_filter, X == 1) >> P(op_get_or, -1))(1)
1
>>> (F(op_filter, X > 1) >> P(op_get_or, -1))(1)
-1
>>> (F(op_filter, X == 1) >> P(op_get_or_call, F(add, 0, -1)))(1)
1
>>> (F(op_filter, X > 1) >> P(op_get_or_call, F(add, 0, -1)))(1)
-1
>>> (F(op_filter, X == 1) >> P(op_map, X + 1) >> P(op_get_or, -1))(1)
2
>>> (F(op_filter, X > 1) >> P(op_map, X + 1) >> P(op_get_or, -1))(1)
-1
>>> (F(op_filter, X == 1) >> P(op_or_else, 2) >> P(op_get_or, -1))(1)
1
>>> (F(op_filter, X > 1) >> P(op_or_else, 2) >> P(op_get_or, -1))(1)
2
>>> (F(op_filter, X == 1) >> P(op_or_call, F(add, 1, 1)) >>
...  P(op_get_or, -1))(1)
1
>>> (F(op_filter, X > 1) >> P(op_or_call, F(add, 1, 1)) >>
...  P(op_get_or, -1))(1)
2
```


### Either like op, but support Exception.

```python
>>> from functoolsex import F, X, P
>>> from operator import add
>>> from toolz import excepts
>>> (F(e_right) >> P(e_filter, X == 1) >> P(e_get_or, -1))(1)
1
>>> (F(e_filter, X > 1) >> P(e_get_or, -1))(e_right(1))
-1
>>> (F(e_right) >> P(e_filter, X == 1) >> P(e_get_or_call, F(add, 0, -1)))(1)
1
>>> (F(e_right) >> P(e_filter, X > 1) >> P(e_get_or_call, F(add, 0, -1)))(1)
-1
>>> (F(e_right) >> P(e_filter, X == 1) >>
...    P(e_map, excepts(ZeroDivisionError, F(1 // X) >> e_right, e_left)) >> P(e_get_or, -1))(1)
1
>>> (F(e_right) >> P(e_filter, X == 1) >>
...    P(e_map, excepts(ZeroDivisionError, F(1 // X) >> e_right, e_left)) >> P(e_get_or, -1))(0)
-1
>>> (excepts(ZeroDivisionError, (F(e_right) >> P(e_filter, X == 0) >>
...    P(e_map, excepts(ZeroDivisionError, F(1 // X) >> e_right, e_left)) >> P(e_get_or_raise)), str)(0) ==
... 'integer division or modulo by zero')
True
>>> (F(e_right) >> P(e_filter, X == 1) >> e_get_right)(1)
1
>>> (excepts(ValueError, (F(e_right) >> P(e_filter, X > 1) >> e_get_right), str)(1) == "('__functoolsex__e__left', None) is not either right")
True
>>> ((F(e_right) >> P(e_filter, X > 1) >> P(e_get_left))(1)) is None
True
>>> (F(e_right) >> P(e_filter, X == 1) >> P(e_or_else, e_right(2)) >> P(e_get_or, -1))(1)
1
>>> (F(e_right) >> P(e_filter, X > 1) >> P(e_or_else, e_right(2)) >> P(e_get_or, -1))(1)
2
>>> (F(e_right) >> P(e_filter, X == 1) >> P(e_or_call, (F(add, 1, 1) >> e_right)) >>
...  P(e_get_or, -1))(1)
1
>>> (F(e_right) >> P(e_filter, X > 1) >> P(e_or_call, (F(add, 1, 1) >> e_right)) >>
...  P(e_get_or, -1))(1)
2
```

### log, very useful to debug.
```python
>>> from operator import add, mul
>>> (F(add, 1) >> log('add res: %s') >> P(mul, 3))(2)
add res: 3
9
```

### fold, like [toolz.sandbox.fold](https://github.com/pytoolz/toolz/blob/ea3ba0d60a33b256c8b2a7be43aff926992ffcdb/toolz/sandbox/parallel.py#L13), but as fast as reduce on PyPy.
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

### tco_yield, while is bad, and yield from is terribly slower. More info [fn.recur.tco](https://github.com/kachayev/fn.py#trampolines-decorator).

```python
>>> from functoolsex import tco_yield
>>> def read_line(fp):
...     @tco_yield
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
# doc test
python -m doctest functoolsex/func.py
python -m doctest functoolsex/iters.py
python -m doctest functoolsex/recur.py

# edit tag in setup.py
git tag v0.0.1
rm dist/functoolsex-*
python setup.py sdist bdist_wheel
twine upload dist/*
```
