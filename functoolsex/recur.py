# -*- coding: utf-8 -*
from __future__ import absolute_import, division, print_function, unicode_literals  # NOQA

__all__ = ('tco_yield', )


class tco_yield(object):
    """Conv function return to yield

    Such function should return one of:
     * None - will exit from loop and stop iter
     * (None, result, args) or (None, result, args, kwargs)
     *  - will yield the result and repeat loop with the same
          function and other arguments
     * (func, result, args) or (func, result, args, kwargs)
     *  - will yield the result and repeat loop
     *    with new callable and new arguments

    Usage example:

    def read_line(fp):
        @tco_yield
        def go(i):
            line = fp.readline()
            if line:
                return None, f"{line[:-1]} {i}", (i + 1, )
        return go(0)

    with open('tmp.txt', 'r') as fp:
        print(list(read_line(fp)))
    """
    __slots__ = "func",

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        action = self
        while True:
            result = action.func(*args, **kwargs)
            if result is None:
                break
            yield result[1]
            act = result[0]
            if callable(act):
                action = act
            args = result[2] if len(result) > 2 else ()
            kwargs = result[3] if len(result) > 3 else {}
