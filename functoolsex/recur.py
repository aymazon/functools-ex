# -*- coding: utf-8 -*
from __future__ import absolute_import, division, print_function, unicode_literals  # NOQA

__all__ = ('tco_yield', 'tco')


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


class tco(object):
    """Provides a trampoline for functions that need one.

    Such function should return one of:
     * (False, result) - will exit from loop and return result to caller
     * (True, args) or (True, args, kwargs) - will repeat loop with the same
                                              function and other arguments
     * (func, args) or (func, kwargs) - will repeat loop with new callable
                                        and new arguments

    Usage example:

    @recur.tco
    def accumulate(origin, f=operator.add, acc=0):
        n = next(origin, None)
        if n is None: return False, acc
        return True, (origin, f, f(acc, n))

    Idea was described on python mailing list:
    http://mail.python.org/pipermail/python-ideas/2009-May/004486.html
    """

    __slots__ = "func",

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        action = self
        while True:
            result = action.func(*args, **kwargs)
            # return final result
            if not result[0]:
                return result[1]

            # next loop with other arguments
            act, args = result[:2]
            # it's possible to given other function to run
            # XXX: do I need to raise exception if it's
            # impossible to use such function in tail calls loop?
            if callable(act):
                action = act
            kwargs = result[2] if len(result) > 2 else {}
