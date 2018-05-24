# -*- coding: utf-8 -*
""" 补充扩展 fn 和 toolz 中没有的迭代工具库 """

from __future__ import absolute_import, division, print_function, unicode_literals  # NOQA

import heapq
import operator
from itertools import cycle, repeat, chain, dropwhile, takewhile, islice, \
    starmap, tee, product, permutations, combinations
from pyrsistent import PList, PVector
from toolz.compatibility import map, filterfalse, filter, range, \
    zip, zip_longest
from toolz.itertoolz import no_default, remove, accumulate, merge_sorted, \
    interleave, unique, take, tail, drop, take_nth, rest, concat, concatv, \
    mapcat, cons, interpose, sliding_window, partition, no_pad, \
    partition_all, pluck, diff, random_sample
from fn.monad import Empty
from fn.iters import compact, reject, ncycles, repeatfunc, grouper, \
    roundrobin, partition as splitin, splitat, splitby, powerset, pairwise, \
    iter_except, flatten

from functoolsex.func import is_not_none, is_option_full

try:
    # 兼容 2.6.6 (>= centos 6.5)
    from itertools import compress, combinations_with_replacement
except ImportError:

    def compress(data, selectors):
        # compress('ABCDEF', [1,0,1,0,1,1]) --> A C E F
        return (d for d, s in zip(data, selectors) if s)

    def combinations_with_replacement(iterable, r):
        # combinations_with_replacement('ABC', 2) --> AA AB AC BB BC CC
        pool = tuple(iterable)
        n = len(pool)
        if not n and r:
            return
        indices = [0] * r
        yield tuple(pool[i] for i in indices)
        while True:
            for i in reversed(range(r)):
                if indices[i] != n - 1:
                    break
            else:
                return
            indices[i:] = [indices[i] + 1] * (r - i)
            yield tuple(pool[i] for i in indices)


__all__ = ("combinations_with_replacement", "compress", "every",
           "first_object", "first_option_full", "first_pred_object",
           "first_true", "getter", "laccumulate", "lchain", "lcombinations",
           "lcombinations_with_replacement", "lcompact", "lcompress",
           "lconcat", "lconcatv", "lcons", "lcycle", "ldiff", "ldrop",
           "ldropwhile", "lfilter", "lfilterfalse", "lflatten", "lgrouper",
           "linterleave", "linterpose", "lislice", "liter_except", "lmap",
           "lmapcat", "lmerge_sorted", "lncycles", "lpairwise", "lpartition",
           "lpartition_all", "lpermutations", "lpluck", "lpowerset",
           "lproduct", "lrandom_sample", "lrange", "lreject", "lremove",
           "lrepeat", "lrepeatfunc", "lrest", "lroundrobin", "lsliding_window",
           "lsplitat", "lsplitby", "lsplitin", "lstarmap", "ltail", "ltake",
           "ltake_nth", "ltakewhile", "ltee", "ltopk", "lunique", "lzip",
           "lzip_longest", "some")


def every(predicate, iterable):
    """
    >>> every(lambda x: x > 1, [2, 0, 3])
    False
    >>> every(lambda x: x >= 0, [2, 0, 3])
    True
    """
    return all(map(predicate, iterable))


def some(predicate, iterable):
    """
    >>> some(lambda x: x > 1, [2, 0, 3])
    True
    >>> some(lambda x: x > 4, [2, 0, 3])
    False
    """
    return any(map(predicate, iterable))


def first_true(iterable, default=False, pred=None):
    """
    >>> first_true([None, 0, {}, 1, True, False])
    1
    """
    return next(filter(pred, iterable), default)


def first_object(iterable, pred=is_not_none):
    """
    >>> first_object([None, 0, {}])
    0
    """
    return first_true(iterable, default=None, pred=pred)


def first_option_full(iterable, pred=is_option_full):
    """
    >>> from fn.monad import Full, Empty
    >>> first_option_full([Empty(), Full(0)])
    Full(0)
    >>> first_option_full([])
    Empty()
    """
    return first_true(iterable, default=Empty(), pred=pred)


def first_pred_object(iterable, pred=is_not_none):
    """
    >>> from toolz.itertoolz import count
    >>> iter = repeat(1, 3)
    >>> first_pred_object(iter, lambda x: "not None")
    u'not None'
    >>> count(iter)
    2
    >>> iter = repeat(1, 3)
    >>> first_pred_object(iter, lambda x: None) is None
    True
    >>> count(iter)
    0
    """
    for i in iterable:
        obj = pred(i)
        if obj is not None:
            return obj
    return None


def lremove(predicate, seq):
    return list(remove(predicate, seq))


def laccumulate(binop, seq, initial=no_default):
    """
    >>> lremove(lambda x: x % 2 == 0, [1, 2, 3, 4])
    [1, 3]
    """
    return list(accumulate(binop, seq, initial=initial))


def lmerge_sorted(*seqs, **kwargs):
    """
    >>> lmerge_sorted([1, 3, 5], [2, 4, 6])
    [1, 2, 3, 4, 5, 6]
    >>> lmerge_sorted([2, 3], [1, 3], key=lambda x: x // 3)
    [2, 1, 3, 3]
    """
    return list(merge_sorted(*seqs, **kwargs))


def linterleave(seqs):
    """
    >>> linterleave([[1, 2], [3, 4]])
    [1, 3, 2, 4]
    """
    return list(interleave(seqs))


def lunique(seq, key=None):
    """
    >>> lunique((1, 2, 3))
    [1, 2, 3]
    """
    return list(unique(seq, key=key))


def ltake(n, seq):
    """
    >>> ltake(2, [10, 20, 30, 40, 50])
    [10, 20]
    """
    return list(take(n, seq))


def ltail(n, seq):
    """ warn tail function
    >>> ltail(2, (10, 20, 30, 40, 50))
    [40, 50]
    """
    return list(tail(n, seq))


def ldrop(n, seq):
    """
    >>> ldrop(2, (10, 20, 30, 40, 50))
    [30, 40, 50]
    """
    return list(drop(n, seq))


def lrest(seq):
    """
    >>> lrest((10, 20, 30, 40, 50))
    [20, 30, 40, 50]
    """
    return list(rest(seq))


def ltake_nth(n, seq):
    """
    >>> list(take_nth(2, (10, 20, 30, 40, 50)))
    [10, 30, 50]
    """
    return list(take_nth(n, seq))


def lconcat(seqs):
    """
    >>> lconcat([[], [1], (2, 3)])
    [1, 2, 3]
    """
    return list(concat(seqs))


def lconcatv(*seqs):
    """
    >>> lconcatv([], [1], (2, 3))
    [1, 2, 3]
    """
    return list(concatv(*seqs))


def lmapcat(func, seqs):
    """
    >>> lmapcat(lambda s: [c.upper() for c in s], \
                [[u"a", u"b"], [u"c", u"d", u"e"]])
    [u'A', u'B', u'C', u'D', u'E']
    """
    return list(mapcat(func, seqs))


def lcons(el, seq):
    """
    >>> lcons(1, (2, 3))
    [1, 2, 3]
    """
    return list(cons(el, seq))


def linterpose(el, seq):
    """
    >>> linterpose(u"a", (1, 2, 3))
    [1, u'a', 2, u'a', 3]
    """
    return list(interpose(el, seq))


def lsliding_window(n, seq):
    """
    >>> lsliding_window(2, [1, 2, 3, 4])
    [(1, 2), (2, 3), (3, 4)]
    """
    return list(sliding_window(n, seq))


def lpartition(n, seq, pad=no_pad):
    """
    >>> lpartition(2, [1, 2, 3, 4])
    [(1, 2), (3, 4)]
    >>> lpartition(2, [1, 2, 3, 4, 5])
    [(1, 2), (3, 4)]
    >>> lpartition(2, [1, 2, 3, 4, 5], pad=None)
    [(1, 2), (3, 4), (5, None)]
    """
    return list(partition(n, seq, pad=pad))


def lpartition_all(n, seq):
    """
    >>> lpartition_all(2, [1, 2, 3, 4])
    [(1, 2), (3, 4)]
    >>> lpartition_all(2, [1, 2, 3, 4, 5])
    [(1, 2), (3, 4), (5,)]
    """
    return list(partition_all(n, seq))


def lpluck(ind, seqs, default=no_default):
    """
    >>> data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    >>> lpluck('name', data)
    [u'Cheese', u'Pies']
    >>> lpluck([0, 1], [[1, 2, 3], [4, 5, 7]])
    [(1, 2), (4, 5)]
    """
    return list(pluck(ind, seqs, default=default))


def ldiff(*seqs, **kwargs):
    """
    >>> ldiff([1, 2, 3], [1, 2, 10, 100])
    [(3, 10)]
    >>> ldiff([1, 2, 3], [1, 2, 10, 100], default=None)
    [(3, 10), (None, 100)]
    >>> ldiff(['apples', 'bananas'], ['Apples', 'Oranges'], key=unicode.lower)
    [(u'bananas', u'Oranges')]
    """
    return list(diff(*seqs, **kwargs))


def getter(index):
    if isinstance(index, list) or isinstance(index, PList) \
            or isinstance(index, PVector):
        if len(index) == 1:
            index = index[0]
            return lambda x: (x[index], )
        elif index:
            return operator.itemgetter(*index)
        else:
            return lambda x: ()
    else:
        return operator.itemgetter(index)


def ltopk(k, seq, key=None):
    """
    >>> ltopk(2, [1, 100, 10, 1000])
    [1000, 100]
    >>> ltopk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len)
    [u'Charlie', u'Alice']
    """
    if key is not None and not callable(key):
        key = getter(key)
    return list(heapq.nlargest(k, seq, key=key))


def lrandom_sample(prob, seq, random_state=None):
    """
    >>> seq = list(range(100))
    >>> lrandom_sample(0.1, seq) # doctest: +SKIP
    [6, 9, 19, 35, 45, 50, 58, 62, 68, 72, 78, 86, 95]
    >>> lrandom_sample(0.1, seq) # doctest: +SKIP
    [6, 44, 54, 61, 69, 94]
    >>> lrandom_sample(0.1, seq, random_state=2016)
    [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]
    >>> lrandom_sample(0.1, seq, random_state=2016)
    [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]
    """
    return list(random_sample(prob, seq, random_state=random_state))


def lrange(*args, **kwargs):
    """
    >>> lrange(3)
    [0, 1, 2]
    >>> lrange(1, 3)
    [1, 2]
    >>> lrange(0, 3, 2)
    [0, 2]
    """
    return list(range(*args, **kwargs))


def lcycle(iterable, n):
    """
    >>> lcycle([1, 2, 3], 2)
    [1, 2]
    >>> lcycle([1, 2, 3], 4)
    [1, 2, 3, 1]
    """

    return list(islice(cycle(iterable), n))


def lrepeat(elem, n):
    """
    >>> lrepeat(1, 2)
    [1, 1]
    """
    return list(repeat(elem, n))


def lchain(*iterables):
    """
    >>> lchain([1, 2], [3, 4])
    [1, 2, 3, 4]
    """
    return list(chain(*iterables))


def lcompress(data, selectors):
    """
    >>> lcompress('ABCDEF', [1, 0, 1, 0, 1, 1])
    [u'A', u'C', u'E', u'F']
    """
    return list(compress(data, selectors))


def ldropwhile(predicate, iterable):
    """
    >>> ldropwhile(lambda x: x < 5 , [1, 4, 6, 4, 1])
    [6, 4, 1]
    """
    return list(dropwhile(predicate, iterable))


def ltakewhile(predicate, iterable):
    """
    >>> ltakewhile(lambda x: x < 5, [1, 4, 6, 4, 1])
    [1, 4]
    """
    return list(takewhile(predicate, iterable))


def lmap(function, *iterables):
    """
    >>> lmap(pow, (2, 3, 10), (5, 2, 3))
    [32, 9, 1000]
    """
    return list(map(function, *iterables))


def lstarmap(function, iterable):
    """
    >>> lstarmap(pow, [(2, 5), (3, 2), (10, 3)])
    [32, 9, 1000]
    """
    return list(starmap(function, iterable))


def ltee(iterable, n=2):
    """
    >>> ltee("ABC")
    [(u'A', u'B', u'C'), (u'A', u'B', u'C')]
    """
    return list(map(tuple, tee(iterable, n)))


def lfilter(predicate, iterable):
    """
    >>> lfilter(lambda x: x % 2, range(10))
    [1, 3, 5, 7, 9]
    """
    return list(filter(predicate, iterable))


def lfilterfalse(predicate, iterable):
    """
    >>> lfilterfalse(lambda x: x % 2, range(10))
    [0, 2, 4, 6, 8]
    """
    return list(filterfalse(predicate, iterable))


def lislice(iterable, *args):
    """ (iterable, stop) or (iterable, start, stop[, step])
    >>> lislice('ABCDEFG', 2)
    [u'A', u'B']
    >>> lislice('ABCDEFG', 2, 4)
    [u'C', u'D']
    >>> lislice('ABCDEFG', 2, None)
    [u'C', u'D', u'E', u'F', u'G']
    >>> lislice('ABCDEFG', 0, None, 2)
    [u'A', u'C', u'E', u'G']
    """
    return list(islice(iterable, *args))


def lzip(*iterables):
    """
    >>> lzip('ABCD', 'xy')
    [(u'A', u'x'), (u'B', u'y')]
    """
    return list(zip(*iterables))


def lzip_longest(*args, **kwds):
    """
    >>> lzip_longest('ABCD', 'xy', fillvalue='-')
    [(u'A', u'x'), (u'B', u'y'), (u'C', u'-'), (u'D', u'-')]
    >>> lzip_longest('ABCD', 'xy')
    [(u'A', u'x'), (u'B', u'y'), (u'C', None), (u'D', None)]
    """
    return list(zip_longest(*args, **kwds))


def lproduct(*args, **kwds):
    """
    >>> lproduct('ABC', 'xy')
    [(u'A', u'x'), (u'A', u'y'), (u'B', u'x'), (u'B', u'y'), (u'C', u'x'), (u'C', u'y')]
    """
    return list(product(*args, **kwds))


def lpermutations(iterable, r=None):
    """
    >>> lpermutations('ABC')
    [(u'A', u'B', u'C'), (u'A', u'C', u'B'), (u'B', u'A', u'C'), (u'B', u'C', u'A'), (u'C', u'A', u'B'), (u'C', u'B', u'A')]
    >>> lpermutations('ABC', 2)
    [(u'A', u'B'), (u'A', u'C'), (u'B', u'A'), (u'B', u'C'), (u'C', u'A'), (u'C', u'B')]
    >>> lpermutations('ABC', 3)
    [(u'A', u'B', u'C'), (u'A', u'C', u'B'), (u'B', u'A', u'C'), (u'B', u'C', u'A'), (u'C', u'A', u'B'), (u'C', u'B', u'A')]
    """
    return list(permutations(iterable, r))


def lcombinations(iterable, r):
    """
    >>> lcombinations('ABCD', 0)
    [()]
    >>> lcombinations('ABCD', 1)
    [(u'A',), (u'B',), (u'C',), (u'D',)]
    >>> lcombinations('ABCD', 2)
    [(u'A', u'B'), (u'A', u'C'), (u'A', u'D'), (u'B', u'C'), (u'B', u'D'), (u'C', u'D')]
    >>> lcombinations('ABCD', 3)
    [(u'A', u'B', u'C'), (u'A', u'B', u'D'), (u'A', u'C', u'D'), (u'B', u'C', u'D')]
    >>> lcombinations('ABCD', 4)
    [(u'A', u'B', u'C', u'D')]
    >>> lcombinations('ABCD', 5)
    []
    """
    return list(combinations(iterable, r))


def lcombinations_with_replacement(iterable, r):
    """
    >>> lcombinations_with_replacement('ABCD', 0)
    [()]
    >>> lcombinations_with_replacement('ABCD', 1)
    [(u'A',), (u'B',), (u'C',), (u'D',)]
    >>> lcombinations_with_replacement('ABCD', 2)
    [(u'A', u'A'), (u'A', u'B'), (u'A', u'C'), (u'A', u'D'), (u'B', u'B'), (u'B', u'C'), (u'B', u'D'), (u'C', u'C'), (u'C', u'D'), (u'D', u'D')]
    >>> lcombinations_with_replacement('ABCD', 3)
    [(u'A', u'A', u'A'), (u'A', u'A', u'B'), (u'A', u'A', u'C'), (u'A', u'A', u'D'), (u'A', u'B', u'B'), (u'A', u'B', u'C'), (u'A', u'B', u'D'), (u'A', u'C', u'C'), (u'A', u'C', u'D'), (u'A', u'D', u'D'), (u'B', u'B', u'B'), (u'B', u'B', u'C'), (u'B', u'B', u'D'), (u'B', u'C', u'C'), (u'B', u'C', u'D'), (u'B', u'D', u'D'), (u'C', u'C', u'C'), (u'C', u'C', u'D'), (u'C', u'D', u'D'), (u'D', u'D', u'D')]
    >>> lcombinations_with_replacement('ABCD', 4)
    [(u'A', u'A', u'A', u'A'), (u'A', u'A', u'A', u'B'), (u'A', u'A', u'A', u'C'), (u'A', u'A', u'A', u'D'), (u'A', u'A', u'B', u'B'), (u'A', u'A', u'B', u'C'), (u'A', u'A', u'B', u'D'), (u'A', u'A', u'C', u'C'), (u'A', u'A', u'C', u'D'), (u'A', u'A', u'D', u'D'), (u'A', u'B', u'B', u'B'), (u'A', u'B', u'B', u'C'), (u'A', u'B', u'B', u'D'), (u'A', u'B', u'C', u'C'), (u'A', u'B', u'C', u'D'), (u'A', u'B', u'D', u'D'), (u'A', u'C', u'C', u'C'), (u'A', u'C', u'C', u'D'), (u'A', u'C', u'D', u'D'), (u'A', u'D', u'D', u'D'), (u'B', u'B', u'B', u'B'), (u'B', u'B', u'B', u'C'), (u'B', u'B', u'B', u'D'), (u'B', u'B', u'C', u'C'), (u'B', u'B', u'C', u'D'), (u'B', u'B', u'D', u'D'), (u'B', u'C', u'C', u'C'), (u'B', u'C', u'C', u'D'), (u'B', u'C', u'D', u'D'), (u'B', u'D', u'D', u'D'), (u'C', u'C', u'C', u'C'), (u'C', u'C', u'C', u'D'), (u'C', u'C', u'D', u'D'), (u'C', u'D', u'D', u'D'), (u'D', u'D', u'D', u'D')]
    >>> lcombinations_with_replacement('ABC', 4)
    [(u'A', u'A', u'A', u'A'), (u'A', u'A', u'A', u'B'), (u'A', u'A', u'A', u'C'), (u'A', u'A', u'B', u'B'), (u'A', u'A', u'B', u'C'), (u'A', u'A', u'C', u'C'), (u'A', u'B', u'B', u'B'), (u'A', u'B', u'B', u'C'), (u'A', u'B', u'C', u'C'), (u'A', u'C', u'C', u'C'), (u'B', u'B', u'B', u'B'), (u'B', u'B', u'B', u'C'), (u'B', u'B', u'C', u'C'), (u'B', u'C', u'C', u'C'), (u'C', u'C', u'C', u'C')]
    """
    return list(combinations_with_replacement(iterable, r))


def lcompact(iterable):
    """
    >>> lcompact((0, 1, 2, False, True, None))
    [1, 2, True]
    """
    return list(compact(iterable))


def lreject(predicate, iterable):
    """
    >>> lreject(lambda x: x > 1, [0, 1, 2])
    [0, 1]
    """
    return list(reject(predicate, iterable))


def lncycles(iterable, n):
    """
    >>> lncycles([1, 2, 3], 2)
    [1, 2, 3, 1, 2, 3]
    """
    return list(ncycles(iterable, n))


def lrepeatfunc(func, times=None, *args):
    """
    >>> lrepeatfunc(lambda : 1, times=3)
    [1, 1, 1]
    """
    return list(repeatfunc(func, times=times, *args))


def lgrouper(n, iterable, fillvalue=None):
    """
    >>> lgrouper(2, [1, 2, 3, 4, 5], 0)
    [(1, 2), (3, 4), (5, 0)]
    """
    return list(grouper(n, iterable, fillvalue=fillvalue))


def lroundrobin(*iterables):
    """
    >>> lroundrobin([1, 2, 3], [4], [5, 6])
    [1, 4, 5, 2, 6, 3]
    """
    return list(roundrobin(*iterables))


def lsplitin(pred, iterable):
    """
    >>> lsplitin(lambda x: x % 2 == 0, [1, 2, 3, 4, 5])
    [(1, 3, 5), (2, 4)]
    """
    return list(map(tuple, splitin(pred, iterable)))


def lsplitat(t, iterable):
    """
    >>> lsplitat(2, range(5))
    [(0, 1), (2, 3, 4)]
    """
    return list(map(tuple, splitat(t, iterable)))


def lsplitby(pred, iterable):
    """
    >>> lsplitby(lambda x: x < 1, range(3))
    [(0,), (1, 2)]
    """
    return list(map(tuple, splitby(pred, iterable)))


def lpowerset(iterable):
    """
    >>> lpowerset([1, 2, 3])
    [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    """
    return list(powerset(iterable))


def lpairwise(iterable):
    """
    >>> lpairwise([1, 2, 3, 4])
    [(1, 2), (2, 3), (3, 4)]
    """
    return list(pairwise(iterable))


def liter_except(func, exception, first_=None):
    """
    >>> d = {1: 1, 2: 2, 3: 3}
    >>> liter_except(d.popitem, KeyError)
    [(1, 1), (2, 2), (3, 3)]
    """
    return list(iter_except(func, exception, first_=first_))


def lflatten(items):
    """
    >>> lflatten([1, (2, 3), (4, 5, (6, (7, (8,))))])
    [1, 2, 3, 4, 5, 6, 7, 8]
    """
    return list(flatten(items))
