# -*- coding: utf-8 -*
from __future__ import absolute_import, division, print_function, unicode_literals  # NOQA

from toolz.compatibility import *  # noqa
from functools import partial
from fn.monad import Full, Empty
from fn.func import identity
from toolz.utils import no_default

__all__ = ("flip", "F", "FF", "XClass", "op_filter", "op_map", "op_or_else",
           "op_or_call", "op_get_or", "op_get_or_call", "R", "fold", "is_none",
           "is_not_none", "is_option_full", "is_option_empty", "uppack_args")


def flip(f):
    """It is faster than fn and toolz.
    >>> from operator import sub
    >>> flip(sub)(3, 2)
    -1
    """
    return lambda x, y: f(y, x)


class F(object):
    """It is faster than the same one in fn.
    >>> from functools import partial
    >>> from operator import add
    >>> F(add, 1)(2) == partial(add, 1)(2)
    True
    >>> from operator import add, mul
    >>> (F(add, 1) >> (mul, 3))(2)
    9
    >>> (F(add, 1) << (mul, 3))(2)
    7
    """
    __slots__ = "f",

    def __init__(self, f=identity, *args, **kwargs):
        self.f = partial(f, *args, **kwargs) if args or kwargs else f

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def __rshift__(self, g):
        if not callable(g):
            f, self.f = self.f, lambda *args, **kwargs: g[0](*(g[1:] + (f(
                *args, **kwargs), )))
        else:
            f, self.f = self.f, lambda *args, **kwargs: g(f(*args, **kwargs))
        return self

    def __lshift__(self, g):
        if not callable(g):
            f, self.f = self.f, lambda *args, **kwargs: f(g[0](*(g[1:] + args),
                                                               **kwargs))
        else:
            f, self.f = self.f, lambda *args, **kwargs: f(g(*args, **kwargs))
        return self


def FF(data, *forms):
    """It is faster than the same one, thread_last in toolz.
    >>> from operator import add, mul
    >>> inc = lambda x: x + 1
    >>> FF(2, inc, (mul, 3))
    9
    >>> FF(2, (mul, 3), inc)
    7
    """
    result = data
    for form in forms:
        if not callable(form):
            result = form[0](*(form[1:] + (result, )))
        else:
            result = form(result)
    return result


class XClass(object):
    """It is like _ in fn, but so faster than that.
    It only does these things:
        (XX + 1 + 2 == 4)(1)

        class A(): x = 'a'
        (XX.x.call('upper') == 'A')(A())

        (XX.call('upper').call('lower') == 'a')('A')

        (XX[0][1] == 1)([(0, 1), (2, 3)])
    Never use it like (X + X), it does not work.
    """
    __slots__ = "f",

    def __init__(self, f=None):
        self.f = f

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def __getattr__(self, name):
        if self.f is None:
            return XClass(lambda o: getattr(o, name))
        else:
            f, self.f = self.f, lambda o: getattr(f(o), name)
            return self

    def call(self, name, *args, **kwargs):
        if self.f is None:
            return XClass(lambda o: getattr(o, name)(*args, **kwargs))
        else:
            f, self.f = self.f, lambda o: getattr(f(o), name)(*args, **kwargs)
            return self

    def __getitem__(self, k):
        if self.f is None:
            return XClass(lambda o: o[k])
        else:
            f, self.f = self.f, lambda o: f(o)[k]
            return self

    def __add__(self, b):
        if self.f is None:
            return XClass(lambda a: a + b)
        else:
            f, self.f = self.f, lambda a: f(a) + b
            return self

    def __mul__(self, b):
        if self.f is None:
            return XClass(lambda a: a * b)
        else:
            f, self.f = self.f, lambda a: f(a) * b
            return self

    def __sub__(self, b):
        if self.f is None:
            return XClass(lambda a: a - b)
        else:
            f, self.f = self.f, lambda a: f(a) - b
            return self

    def __mod__(self, b):
        if self.f is None:
            return XClass(lambda a: a % b)
        else:
            f, self.f = self.f, lambda a: f(a) % b
            return self

    def __pow__(self, b):
        if self.f is None:
            return XClass(lambda a: a**b)
        else:
            f, self.f = self.f, lambda a: f(a)**b
            return self

    def __and__(self, b):
        raise NotImplementedError

    def __or__(self, b):
        raise NotImplementedError

    def __xor__(self, b):
        if self.f is None:
            return XClass(lambda a: a ^ b)
        else:
            f, self.f = self.f, lambda a: f(a) ^ b
            return self

    def __divmod__(self, b):
        raise NotImplementedError

    def __floordiv__(self, b):
        if self.f is None:
            return XClass(lambda a: a // b)
        else:
            f, self.f = self.f, lambda a: f(a) // b
            return self

    def __truediv__(self, b):
        if self.f is None:
            return XClass(lambda a: a / b)
        else:
            f, self.f = self.f, lambda a: f(a) / b
            return self

    def __lshift__(self, b):
        if self.f is None:
            return XClass(lambda a: a << b)
        else:
            f, self.f = self.f, lambda a: f(a) << b
            return self

    def __rshift__(self, b):
        if self.f is None:
            return XClass(lambda a: a >> b)
        else:
            f, self.f = self.f, lambda a: f(a) >> b
            return self

    def __lt__(self, b):
        if self.f is None:
            return XClass(lambda a: a < b)
        else:
            f, self.f = self.f, lambda a: f(a) < b
            return self

    def __le__(self, b):
        if self.f is None:
            return XClass(lambda a: a <= b)
        else:
            f, self.f = self.f, lambda a: f(a) <= b
            return self

    def __gt__(self, b):
        if self.f is None:
            return XClass(lambda a: a > b)
        else:
            f, self.f = self.f, lambda a: f(a) > b
            return self

    def __ge__(self, b):
        if self.f is None:
            return XClass(lambda a: a >= b)
        else:
            f, self.f = self.f, lambda a: f(a) >= b
            return self

    def __eq__(self, b):
        if self.f is None:
            return XClass(lambda a: a == b)
        else:
            f, self.f = self.f, lambda a: f(a) == b
            return self

    def __ne__(self, b):
        if self.f is None:
            return XClass(lambda a: a != b)
        else:
            f, self.f = self.f, lambda a: f(a) != b
            return self

    def __neg__(self):
        if self.f is None:
            return XClass(lambda a: -a)
        else:
            f, self.f = self.f, lambda a: -f(a)
            return self

    def __pos__(self):
        if self.f is None:
            return XClass(lambda a: +a)
        else:
            f, self.f = self.f, lambda a: +f(a)
            return self

    def __invert__(self):
        if self.f is None:
            return XClass(lambda a: ~a)
        else:
            f, self.f = self.f, lambda a: ~f(a)
            return self

    def __radd__(self, a):
        if self.f is None:
            return XClass(lambda b: a + b)
        else:
            f, self.f = self.f, lambda b: a + f(b)
            return self

    def __rmul__(self, a):
        if self.f is None:
            return XClass(lambda b: a * b)
        else:
            f, self.f = self.f, lambda b: a * f(b)
            return self

    def __rsub__(self, a):
        if self.f is None:
            return XClass(lambda b: a - b)
        else:
            f, self.f = self.f, lambda b: a - f(b)
            return self

    def __rmod__(self, a):
        if self.f is None:
            return XClass(lambda b: a % b)
        else:
            f, self.f = self.f, lambda b: a % f(b)
            return self

    def __rpow__(self, a):
        if self.f is None:
            return XClass(lambda b: a**b)
        else:
            f, self.f = self.f, lambda b: a**f(b)
            return self

    def __rand__(self, a):
        if self.f is None:
            return XClass(lambda b: a and b)
        else:
            f, self.f = self.f, lambda b: a and f(b)
            return self

    def __ror__(self, a):
        if self.f is None:
            return XClass(lambda b: a or b)
        else:
            f, self.f = self.f, lambda b: a or f(b)
            return self

    def __rxor__(self, a):
        if self.f is None:
            return XClass(lambda b: a ^ b)
        else:
            f, self.f = self.f, lambda b: a ^ f(b)
            return self

    def __rdivmod__(self, a):
        raise NotImplementedError

    def __rfloordiv__(self, a):
        if self.f is None:
            return XClass(lambda b: a // b)
        else:
            f, self.f = self.f, lambda b: a // f(b)
            return self

    def __rtruediv__(self, a):
        if self.f is None:
            return XClass(lambda b: a / b)
        else:
            f, self.f = self.f, lambda b: a / f(b)
            return self

    def __rlshift__(self, a):
        if self.f is None:
            return XClass(lambda b: a << b)
        else:
            f, self.f = self.f, lambda b: a << f(b)
            return self

    def __rrshift__(self, a):
        if self.f is None:
            return XClass(lambda b: a >> b)
        else:
            f, self.f = self.f, lambda b: a >> f(b)
            return self


class R(object):
    """Run functions with the same args, support tuple partial.
    >>> from functoolsex import X
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
    """
    __slots__ = "fs",

    def __init__(self, *forms):
        self.fs = fs = []
        for f in forms:
            if not callable(f):
                f = partial(f[0], *f[1:])
            fs.append(f)

    def __call__(self, *args, **kwargs):
        result = []
        for f in self.fs:
            result.append(f(*args, **kwargs))
        return tuple(result)


__op_empty = '__functoolsex__op__empty'


def op_filter(func, val):
    """Option filter map and get value, like Option in fn.
    >>> from functoolsex import F, X
    >>> from operator import add
    >>> (F(op_filter, X == 1) >> (op_get_or, -1))(1)
    1
    >>> (F(op_filter, X > 1) >> (op_get_or, -1))(1)
    -1
    >>> (F(op_filter, X == 1) >> (op_get_or_call, F(add, 0, -1)))(1)
    1
    >>> (F(op_filter, X > 1) >> (op_get_or_call, F(add, 0, -1)))(1)
    -1
    >>> (F(op_filter, X == 1) >> (op_map, X + 1) >> (op_get_or, -1))(1)
    2
    >>> (F(op_filter, X > 1) >> (op_map, X + 1) >> (op_get_or, -1))(1)
    -1
    >>> (F(op_filter, X == 1) >> (op_or_else, 2) >> (op_get_or, -1))(1)
    1
    >>> (F(op_filter, X > 1) >> (op_or_else, 2) >> (op_get_or, -1))(1)
    2
    >>> (F(op_filter, X == 1) >> (op_or_call, F(add, 1, 1)) >>
    ...  (op_get_or, -1))(1)
    1
    >>> (F(op_filter, X > 1) >> (op_or_call, F(add, 1, 1)) >>
    ...  (op_get_or, -1))(1)
    2
    """
    return val if val != __op_empty and func(val) else __op_empty


def op_map(func, val):
    return func(val) if val != __op_empty else __op_empty


def op_or_else(else_val, val):
    return else_val if val == __op_empty else val


def op_or_call(func, val):
    return func() if val == __op_empty else val


def op_get_or(default, val):
    return default if val == __op_empty else val


def op_get_or_call(func, val):
    return func() if val == __op_empty else val


def fold(binop, seq, default=no_default, map=map, pool_size=0, combine=None):
    """It is faster than toolz, can take the place of reduce.
    >>> from operator import add
    >>> fold(add, range(10))
    45
    >>> fold(add, range(10), 1)
    46
    >>> from multiprocessing import Pool, cpu_count
    >>> pool_map, pool_size = Pool().imap, cpu_count()
    >>> fold(add, range(1000), map=pool_map, pool_size=pool_size)
    499500
    >>> fold(add, range(1000), 1, map=pool_map, pool_size=pool_size)
    499501
    """
    if combine is None:
        combine = binop
    if default is no_default:
        if not isinstance(seq, list):
            results = list(seq)
        else:
            results = seq
    else:
        results = [default]
        results.extend(seq)
    is_combine = False
    while True:
        chunks = []
        results_len = len(results)
        chunksize = results_len // (pool_size + 1)
        chunksize = 2 if chunksize < pool_size else chunksize
        for i in range(0, results_len, chunksize):
            chunks.append(results[i:i + chunksize])
        if not is_combine:
            results = list(map(partial(reduce, binop), chunks))
            is_combine = True
        else:
            results = list(map(partial(reduce, combine), chunks))
        if len(results) == 1:
            return results[0]


def is_none(obj):
    return obj is None


def is_not_none(obj):
    return obj is not None


def is_option_full(obj):
    return isinstance(obj, Full)


def is_option_empty(obj):
    return isinstance(obj, Empty)


def uppack_args(func, args):
    """Up pack a tuple into function like *args.
    >>> from collections import namedtuple
    >>> from functoolsex import F, X
    >>> from toolz import juxt
    >>> Attr = namedtuple("Attr", ("name", "value"))
    >>> parse_name = F(X.call("partition", "=")[0])
    >>> parse_value = F(X.call("partition", "=")[2])
    >>> load = F(juxt(parse_name, parse_value)) >> F(uppack_args, Attr)
    >>> load("a=b")
    Attr(name='a', value='b')
    """
    return func(*args)
