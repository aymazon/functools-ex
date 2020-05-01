# -*- coding: utf-8 -*
""" 补充扩展 fn 和 toolz 中没有的迭代工具库 """

from __future__ import absolute_import, division, print_function, unicode_literals  # NOQA

import platform
import heapq
import operator
from itertools import cycle, repeat, chain, dropwhile, takewhile, islice, \
    starmap, tee, product, permutations, combinations
from pyrsistent import PList, PVector
from toolz.compatibility import *  # noqa
from toolz.itertoolz import (no_default, remove, accumulate, merge_sorted,
                             interleave, unique, take, tail, drop, take_nth,
                             rest, concat, concatv, mapcat, cons, interpose,
                             sliding_window, partition, no_pad, partition_all,
                             pluck, join, diff, random_sample)
from toolz import identity, partitionby

try:
    try:
        import cytoolz  # noqa
    except ImportError:
        pass
    else:
        if platform.python_implementation() == 'CPython':
            from cytoolz.itertoolz import (  # noqa
                no_default, remove, accumulate, merge_sorted, interleave,
                unique, take, tail, drop, take_nth, rest, concat, concatv,
                mapcat, cons, interpose, sliding_window, partition, no_pad,
                partition_all, pluck, join, diff, random_sample)
            from cytoolz import identity, partitionby  # noqa
except Exception:
    pass
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
           "lpartition_all", "lpermutations", "lpluck", "ljoin", "lpowerset",
           "lproduct", "lrandom_sample", "lpartitionby", "lrange", "lreject",
           "lremove", "lrepeat", "lrepeatfunc", "lrest", "lroundrobin",
           "lsliding_window", "lsplitat", "lsplitby", "lsplitin", "lstarmap",
           "ltail", "ltake", "ltake_nth", "ltakewhile", "ltee", "ltopk",
           "lunique", "lzip", "lzip_longest", "taccumulate", "tchain",
           "tcombinations", "tcombinations_with_replacement", "tcompact",
           "tcompress", "tconcat", "tconcatv", "tcons", "tcycle", "tdiff",
           "tdrop", "tdropwhile", "tfilter", "tfilterfalse", "tflatten",
           "tgrouper", "tinterleave", "tinterpose", "tislice", "titer_except",
           "tmap", "tmapcat", "tmerge_sorted", "tncycles", "tpairwise",
           "tpartition", "tpartition_all", "tpermutations", "tpluck", "tjoin",
           "tpowerset", "tproduct", "trandom_sample", "tpartitionby", "trange",
           "treject", "tremove", "trepeat", "trepeatfunc", "trest",
           "troundrobin", "tsliding_window", "tsplitat", "tsplitby",
           "tsplitin", "tstarmap", "ttail", "ttake", "ttake_nth", "ttakewhile",
           "ttee", "ttopk", "tunique", "tzip", "tzip_longest", "some")


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
    'not None'
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


def getter(index):
    if isinstance(index, list) or isinstance(index, tuple) \
            or isinstance(index, PList) or isinstance(index, PVector):
        if len(index) == 1:
            index = index[0]
            return lambda x: (x[index], )
        elif index:
            return operator.itemgetter(*index)
        else:
            return lambda x: ()
    else:
        return operator.itemgetter(index)


def lremove(predicate, seq):
    """
    >>> lremove(lambda x: x % 2 == 0, [1, 2, 3, 4])
    [1, 3]
    """
    return list(remove(predicate, seq))


def laccumulate(binop, seq, initial=no_default):
    """
    >>> laccumulate(lambda x, y: x + y, [1, 2, 3, 4, 5])
    [1, 3, 6, 10, 15]
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
    >>> ltake_nth(2, (10, 20, 30, 40, 50))
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
    ['A', 'B', 'C', 'D', 'E']
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
    [1, 'a', 2, 'a', 3]
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
    ['Cheese', 'Pies']
    >>> lpluck([0, 1], [[1, 2, 3], [4, 5, 7]])
    [(1, 2), (4, 5)]
    """
    return list(pluck(ind, seqs, default=default))


def ljoin(leftkey, leftseq, rightkey, rightseq,
          left_default=no_default, right_default=no_default):
    """
    >>> ljoin(identity, [1, 2, 3], identity, [2, 3, 4],
    ...       left_default=None, right_default=None)
    [(2, 2), (3, 3), (None, 4), (1, None)]
    """
    return list(
        join(leftkey, leftseq, rightkey, rightseq,
             left_default=left_default, right_default=right_default))


def ldiff(*seqs, **kwargs):
    """
    >>> ldiff([1, 2, 3], [1, 2, 10, 100])
    [(3, 10)]
    >>> ldiff([1, 2, 3], [1, 2, 10, 100], default=None)
    [(3, 10), (None, 100)]
    >>> ldiff(['apples', 'bananas'], ['Apples', 'Oranges'], key=str.lower)
    [('bananas', 'Oranges')]
    """
    return list(diff(*seqs, **kwargs))


def ltopk(k, seq, key=None):
    """
    >>> ltopk(2, [1, 100, 10, 1000])
    [1000, 100]
    >>> ltopk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len)
    ['Charlie', 'Alice']
    """
    if key is not None and not callable(key):
        key = getter(key)
    return list(heapq.nlargest(k, seq, key=key))


def lrandom_sample(pred, seq, random_state=None):
    """
    >>> seq = list(range(100))
    >>> lrandom_sample(0.1, seq) # doctest: +SKIP
    [6, 9, 19, 35, 45, 50, 58, 62, 68, 72, 78, 86, 95]
    >>> lrandom_sample(0.1, seq) # doctest: +SKIP
    [6, 44, 54, 61, 69, 94]
    >>> lrandom_sample(0.1, seq, random_state=2016)
    [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]
    """
    return list(random_sample(pred, seq, random_state=random_state))


def lpartitionby(pred, seq):
    """
    >>> is_space = lambda c: c == " "
    >>> lpartitionby(is_space, "I have space")
    [('I',), (' ',), ('h', 'a', 'v', 'e'), (' ',), ('s', 'p', 'a', 'c', 'e')]
    >>> is_large = lambda x: x > 10
    >>> lpartitionby(is_large, [1, 2, 1, 99, 88, 33, 99, -1, 5])
    [(1, 2, 1), (99, 88, 33, 99), (-1, 5)]
    """
    return list(partitionby(pred, seq))


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
    ['A', 'C', 'E', 'F']
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
    [('A', 'B', 'C'), ('A', 'B', 'C')]
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
    ['A', 'B']
    >>> lislice('ABCDEFG', 2, 4)
    ['C', 'D']
    >>> lislice('ABCDEFG', 2, None)
    ['C', 'D', 'E', 'F', 'G']
    >>> lislice('ABCDEFG', 0, None, 2)
    ['A', 'C', 'E', 'G']
    """
    return list(islice(iterable, *args))


def lzip(*iterables):
    """
    >>> lzip('ABCD', 'xy')
    [('A', 'x'), ('B', 'y')]
    """
    return list(zip(*iterables))


def lzip_longest(*args, **kwds):
    """
    >>> lzip_longest('ABCD', 'xy', fillvalue='-')
    [('A', 'x'), ('B', 'y'), ('C', '-'), ('D', '-')]
    >>> lzip_longest('ABCD', 'xy')
    [('A', 'x'), ('B', 'y'), ('C', None), ('D', None)]
    """
    return list(zip_longest(*args, **kwds))


def lproduct(*args, **kwds):
    """
    >>> lproduct('ABC', 'xy')
    [('A', 'x'), ('A', 'y'), ('B', 'x'), ('B', 'y'), ('C', 'x'), ('C', 'y')]
    """
    return list(product(*args, **kwds))


def lpermutations(iterable, r=None):
    """
    >>> lpermutations('ABC')
    [('A', 'B', 'C'), ('A', 'C', 'B'), ('B', 'A', 'C'), ('B', 'C', 'A'), ('C', 'A', 'B'), ('C', 'B', 'A')]
    >>> lpermutations('ABC', 2)
    [('A', 'B'), ('A', 'C'), ('B', 'A'), ('B', 'C'), ('C', 'A'), ('C', 'B')]
    >>> lpermutations('ABC', 3)
    [('A', 'B', 'C'), ('A', 'C', 'B'), ('B', 'A', 'C'), ('B', 'C', 'A'), ('C', 'A', 'B'), ('C', 'B', 'A')]
    """
    return list(permutations(iterable, r))


def lcombinations(iterable, r):
    """
    >>> lcombinations('ABCD', 0)
    [()]
    >>> lcombinations('ABCD', 1)
    [('A',), ('B',), ('C',), ('D',)]
    >>> lcombinations('ABCD', 2)
    [('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'C'), ('B', 'D'), ('C', 'D')]
    >>> lcombinations('ABCD', 3)
    [('A', 'B', 'C'), ('A', 'B', 'D'), ('A', 'C', 'D'), ('B', 'C', 'D')]
    >>> lcombinations('ABCD', 4)
    [('A', 'B', 'C', 'D')]
    >>> lcombinations('ABCD', 5)
    []
    """
    return list(combinations(iterable, r))


def lcombinations_with_replacement(iterable, r):
    """
    >>> lcombinations_with_replacement('ABCD', 0)
    [()]
    >>> lcombinations_with_replacement('ABCD', 1)
    [('A',), ('B',), ('C',), ('D',)]
    >>> lcombinations_with_replacement('ABCD', 2)
    [('A', 'A'), ('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'B'), ('B', 'C'), ('B', 'D'), ('C', 'C'), ('C', 'D'), ('D', 'D')]
    >>> lcombinations_with_replacement('ABCD', 3)
    [('A', 'A', 'A'), ('A', 'A', 'B'), ('A', 'A', 'C'), ('A', 'A', 'D'), ('A', 'B', 'B'), ('A', 'B', 'C'), ('A', 'B', 'D'), ('A', 'C', 'C'), ('A', 'C', 'D'), ('A', 'D', 'D'), ('B', 'B', 'B'), ('B', 'B', 'C'), ('B', 'B', 'D'), ('B', 'C', 'C'), ('B', 'C', 'D'), ('B', 'D', 'D'), ('C', 'C', 'C'), ('C', 'C', 'D'), ('C', 'D', 'D'), ('D', 'D', 'D')]
    >>> lcombinations_with_replacement('ABCD', 4)
    [('A', 'A', 'A', 'A'), ('A', 'A', 'A', 'B'), ('A', 'A', 'A', 'C'), ('A', 'A', 'A', 'D'), ('A', 'A', 'B', 'B'), ('A', 'A', 'B', 'C'), ('A', 'A', 'B', 'D'), ('A', 'A', 'C', 'C'), ('A', 'A', 'C', 'D'), ('A', 'A', 'D', 'D'), ('A', 'B', 'B', 'B'), ('A', 'B', 'B', 'C'), ('A', 'B', 'B', 'D'), ('A', 'B', 'C', 'C'), ('A', 'B', 'C', 'D'), ('A', 'B', 'D', 'D'), ('A', 'C', 'C', 'C'), ('A', 'C', 'C', 'D'), ('A', 'C', 'D', 'D'), ('A', 'D', 'D', 'D'), ('B', 'B', 'B', 'B'), ('B', 'B', 'B', 'C'), ('B', 'B', 'B', 'D'), ('B', 'B', 'C', 'C'), ('B', 'B', 'C', 'D'), ('B', 'B', 'D', 'D'), ('B', 'C', 'C', 'C'), ('B', 'C', 'C', 'D'), ('B', 'C', 'D', 'D'), ('B', 'D', 'D', 'D'), ('C', 'C', 'C', 'C'), ('C', 'C', 'C', 'D'), ('C', 'C', 'D', 'D'), ('C', 'D', 'D', 'D'), ('D', 'D', 'D', 'D')]
    >>> lcombinations_with_replacement('ABC', 4)
    [('A', 'A', 'A', 'A'), ('A', 'A', 'A', 'B'), ('A', 'A', 'A', 'C'), ('A', 'A', 'B', 'B'), ('A', 'A', 'B', 'C'), ('A', 'A', 'C', 'C'), ('A', 'B', 'B', 'B'), ('A', 'B', 'B', 'C'), ('A', 'B', 'C', 'C'), ('A', 'C', 'C', 'C'), ('B', 'B', 'B', 'B'), ('B', 'B', 'B', 'C'), ('B', 'B', 'C', 'C'), ('B', 'C', 'C', 'C'), ('C', 'C', 'C', 'C')]
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
    [(3, 3), (2, 2), (1, 1)]
    """
    return list(iter_except(func, exception, first_=first_))


def lflatten(items):
    """
    >>> lflatten([1, (2, 3), (4, 5, (6, (7, (8,))))])
    [1, 2, 3, 4, 5, 6, 7, 8]
    """
    return list(flatten(items))


def tremove(predicate, seq):
    """
    >>> tremove(lambda x: x % 2 == 0, [1, 2, 3, 4])
    (1, 3)
    """
    return tuple(remove(predicate, seq))


def taccumulate(binop, seq, initial=no_default):
    """
    >>> taccumulate(lambda x, y: x + y, [1, 2, 3, 4, 5])
    (1, 3, 6, 10, 15)
    """
    return tuple(accumulate(binop, seq, initial=initial))


def tmerge_sorted(*seqs, **kwargs):
    """
    >>> tmerge_sorted([1, 3, 5], [2, 4, 6])
    (1, 2, 3, 4, 5, 6)
    >>> tmerge_sorted([2, 3], [1, 3], key=lambda x: x // 3)
    (2, 1, 3, 3)
    """
    return tuple(merge_sorted(*seqs, **kwargs))


def tinterleave(seqs):
    """
    >>> tinterleave([[1, 2], [3, 4]])
    (1, 3, 2, 4)
    """
    return tuple(interleave(seqs))


def tunique(seq, key=None):
    """
    >>> tunique((1, 2, 3))
    (1, 2, 3)
    """
    return tuple(unique(seq, key=key))


def ttake(n, seq):
    """
    >>> ttake(2, [10, 20, 30, 40, 50])
    (10, 20)
    """
    return tuple(take(n, seq))


def ttail(n, seq):
    """ warn tail function
    >>> ttail(2, (10, 20, 30, 40, 50))
    (40, 50)
    """
    return tuple(tail(n, seq))


def tdrop(n, seq):
    """
    >>> tdrop(2, (10, 20, 30, 40, 50))
    (30, 40, 50)
    """
    return tuple(drop(n, seq))


def trest(seq):
    """
    >>> trest((10, 20, 30, 40, 50))
    (20, 30, 40, 50)
    """
    return tuple(rest(seq))


def ttake_nth(n, seq):
    """
    >>> ttake_nth(2, (10, 20, 30, 40, 50))
    (10, 30, 50)
    """
    return tuple(take_nth(n, seq))


def tconcat(seqs):
    """
    >>> tconcat([[], [1], (2, 3)])
    (1, 2, 3)
    """
    return tuple(concat(seqs))


def tconcatv(*seqs):
    """
    >>> tconcatv([], [1], (2, 3))
    (1, 2, 3)
    """
    return tuple(concatv(*seqs))


def tmapcat(func, seqs):
    """
    >>> tmapcat(lambda s: [c.upper() for c in s], \
                [[u"a", u"b"], [u"c", u"d", u"e"]])
    ('A', 'B', 'C', 'D', 'E')
    """
    return tuple(mapcat(func, seqs))


def tcons(el, seq):
    """
    >>> tcons(1, (2, 3))
    (1, 2, 3)
    """
    return tuple(cons(el, seq))


def tinterpose(el, seq):
    """
    >>> tinterpose(u"a", (1, 2, 3))
    (1, 'a', 2, 'a', 3)
    """
    return tuple(interpose(el, seq))


def tsliding_window(n, seq):
    """
    >>> tsliding_window(2, [1, 2, 3, 4])
    ((1, 2), (2, 3), (3, 4))
    """
    return tuple(sliding_window(n, seq))


def tpartition(n, seq, pad=no_pad):
    """
    >>> tpartition(2, [1, 2, 3, 4])
    ((1, 2), (3, 4))
    >>> tpartition(2, [1, 2, 3, 4, 5])
    ((1, 2), (3, 4))
    >>> tpartition(2, [1, 2, 3, 4, 5], pad=None)
    ((1, 2), (3, 4), (5, None))
    """
    return tuple(partition(n, seq, pad=pad))


def tpartition_all(n, seq):
    """
    >>> tpartition_all(2, [1, 2, 3, 4])
    ((1, 2), (3, 4))
    >>> tpartition_all(2, [1, 2, 3, 4, 5])
    ((1, 2), (3, 4), (5,))
    """
    return tuple(partition_all(n, seq))


def tpluck(ind, seqs, default=no_default):
    """
    >>> data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    >>> tpluck('name', data)
    ('Cheese', 'Pies')
    >>> tpluck([0, 1], [[1, 2, 3], [4, 5, 7]])
    ((1, 2), (4, 5))
    """
    return tuple(pluck(ind, seqs, default=default))


def tjoin(leftkey, leftseq, rightkey, rightseq,
          left_default=no_default, right_default=no_default):
    """
    >>> tjoin(identity, [1, 2, 3], identity, [2, 3, 4],
    ...       left_default=None, right_default=None)
    ((2, 2), (3, 3), (None, 4), (1, None))
    """
    return tuple(
        join(leftkey, leftseq, rightkey, rightseq,
             left_default=left_default, right_default=right_default))


def tdiff(*seqs, **kwargs):
    """
    >>> tdiff([1, 2, 3], [1, 2, 10, 100])
    ((3, 10),)
    >>> tdiff([1, 2, 3], [1, 2, 10, 100], default=None)
    ((3, 10), (None, 100))
    >>> tdiff(['apples', 'bananas'], ['Apples', 'Oranges'], key=str.lower)
    (('bananas', 'Oranges'),)
    """
    return tuple(diff(*seqs, **kwargs))


def ttopk(k, seq, key=None):
    """
    >>> ttopk(2, [1, 100, 10, 1000])
    (1000, 100)
    >>> ttopk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len)
    ('Charlie', 'Alice')
    """
    if key is not None and not callable(key):
        key = getter(key)
    return tuple(heapq.nlargest(k, seq, key=key))


def trandom_sample(pred, seq, random_state=None):
    """
    >>> seq = list(range(100))
    >>> trandom_sample(0.1, seq) # doctest: +SKIP
    (6, 9, 19, 35, 45, 50, 58, 62, 68, 72, 78, 86, 95)
    >>> trandom_sample(0.1, seq) # doctest: +SKIP
    (6, 44, 54, 61, 69, 94)
    >>> trandom_sample(0.1, seq, random_state=2016)
    (7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98)
    """
    return tuple(random_sample(pred, seq, random_state=random_state))


def tpartitionby(pred, seq):
    """
    >>> is_space = lambda c: c == " "
    >>> tpartitionby(is_space, "I have space")
    (('I',), (' ',), ('h', 'a', 'v', 'e'), (' ',), ('s', 'p', 'a', 'c', 'e'))
    >>> is_large = lambda x: x > 10
    >>> tpartitionby(is_large, [1, 2, 1, 99, 88, 33, 99, -1, 5])
    ((1, 2, 1), (99, 88, 33, 99), (-1, 5))
    """
    return tuple(partitionby(pred, seq))


def trange(*args, **kwargs):
    """
    >>> trange(3)
    (0, 1, 2)
    >>> trange(1, 3)
    (1, 2)
    >>> trange(0, 3, 2)
    (0, 2)
    """
    return tuple(range(*args, **kwargs))


def tcycle(iterable, n):
    """
    >>> tcycle([1, 2, 3], 2)
    (1, 2)
    >>> tcycle([1, 2, 3], 4)
    (1, 2, 3, 1)
    """

    return tuple(islice(cycle(iterable), n))


def trepeat(elem, n):
    """
    >>> trepeat(1, 2)
    (1, 1)
    """
    return tuple(repeat(elem, n))


def tchain(*iterables):
    """
    >>> tchain([1, 2], [3, 4])
    (1, 2, 3, 4)
    """
    return tuple(chain(*iterables))


def tcompress(data, selectors):
    """
    >>> tcompress('ABCDEF', [1, 0, 1, 0, 1, 1])
    ('A', 'C', 'E', 'F')
    """
    return tuple(compress(data, selectors))


def tdropwhile(predicate, iterable):
    """
    >>> tdropwhile(lambda x: x < 5 , [1, 4, 6, 4, 1])
    (6, 4, 1)
    """
    return tuple(dropwhile(predicate, iterable))


def ttakewhile(predicate, iterable):
    """
    >>> ttakewhile(lambda x: x < 5, [1, 4, 6, 4, 1])
    (1, 4)
    """
    return tuple(takewhile(predicate, iterable))


def tmap(function, *iterables):
    """
    >>> tmap(pow, (2, 3, 10), (5, 2, 3))
    (32, 9, 1000)
    """
    return tuple(map(function, *iterables))


def tstarmap(function, iterable):
    """
    >>> tstarmap(pow, [(2, 5), (3, 2), (10, 3)])
    (32, 9, 1000)
    """
    return tuple(starmap(function, iterable))


def ttee(iterable, n=2):
    """
    >>> ttee("ABC")
    (('A', 'B', 'C'), ('A', 'B', 'C'))
    """
    return tuple(map(tuple, tee(iterable, n)))


def tfilter(predicate, iterable):
    """
    >>> tfilter(lambda x: x % 2, range(10))
    (1, 3, 5, 7, 9)
    """
    return tuple(filter(predicate, iterable))


def tfilterfalse(predicate, iterable):
    """
    >>> tfilterfalse(lambda x: x % 2, range(10))
    (0, 2, 4, 6, 8)
    """
    return tuple(filterfalse(predicate, iterable))


def tislice(iterable, *args):
    """ (iterable, stop) or (iterable, start, stop[, step])
    >>> tislice('ABCDEFG', 2)
    ('A', 'B')
    >>> tislice('ABCDEFG', 2, 4)
    ('C', 'D')
    >>> tislice('ABCDEFG', 2, None)
    ('C', 'D', 'E', 'F', 'G')
    >>> tislice('ABCDEFG', 0, None, 2)
    ('A', 'C', 'E', 'G')
    """
    return tuple(islice(iterable, *args))


def tzip(*iterables):
    """
    >>> tzip('ABCD', 'xy')
    (('A', 'x'), ('B', 'y'))
    """
    return tuple(zip(*iterables))


def tzip_longest(*args, **kwds):
    """
    >>> tzip_longest('ABCD', 'xy', fillvalue='-')
    (('A', 'x'), ('B', 'y'), ('C', '-'), ('D', '-'))
    >>> tzip_longest('ABCD', 'xy')
    (('A', 'x'), ('B', 'y'), ('C', None), ('D', None))
    """
    return tuple(zip_longest(*args, **kwds))


def tproduct(*args, **kwds):
    """
    >>> tproduct('ABC', 'xy')
    (('A', 'x'), ('A', 'y'), ('B', 'x'), ('B', 'y'), ('C', 'x'), ('C', 'y'))
    """
    return tuple(product(*args, **kwds))


def tpermutations(iterable, r=None):
    """
    >>> tpermutations('ABC')
    (('A', 'B', 'C'), ('A', 'C', 'B'), ('B', 'A', 'C'), ('B', 'C', 'A'), ('C', 'A', 'B'), ('C', 'B', 'A'))
    >>> tpermutations('ABC', 2)
    (('A', 'B'), ('A', 'C'), ('B', 'A'), ('B', 'C'), ('C', 'A'), ('C', 'B'))
    >>> tpermutations('ABC', 3)
    (('A', 'B', 'C'), ('A', 'C', 'B'), ('B', 'A', 'C'), ('B', 'C', 'A'), ('C', 'A', 'B'), ('C', 'B', 'A'))
    """
    return tuple(permutations(iterable, r))


def tcombinations(iterable, r):
    """
    >>> tcombinations('ABCD', 0)
    ((),)
    >>> tcombinations('ABCD', 1)
    (('A',), ('B',), ('C',), ('D',))
    >>> tcombinations('ABCD', 2)
    (('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'C'), ('B', 'D'), ('C', 'D'))
    >>> tcombinations('ABCD', 3)
    (('A', 'B', 'C'), ('A', 'B', 'D'), ('A', 'C', 'D'), ('B', 'C', 'D'))
    >>> tcombinations('ABCD', 4)
    (('A', 'B', 'C', 'D'),)
    >>> tcombinations('ABCD', 5)
    ()
    """
    return tuple(combinations(iterable, r))


def tcombinations_with_replacement(iterable, r):
    """
    >>> tcombinations_with_replacement('ABCD', 0)
    ((),)
    >>> tcombinations_with_replacement('ABCD', 1)
    (('A',), ('B',), ('C',), ('D',))
    >>> tcombinations_with_replacement('ABCD', 2)
    (('A', 'A'), ('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'B'), ('B', 'C'), ('B', 'D'), ('C', 'C'), ('C', 'D'), ('D', 'D'))
    >>> tcombinations_with_replacement('ABCD', 3)
    (('A', 'A', 'A'), ('A', 'A', 'B'), ('A', 'A', 'C'), ('A', 'A', 'D'), ('A', 'B', 'B'), ('A', 'B', 'C'), ('A', 'B', 'D'), ('A', 'C', 'C'), ('A', 'C', 'D'), ('A', 'D', 'D'), ('B', 'B', 'B'), ('B', 'B', 'C'), ('B', 'B', 'D'), ('B', 'C', 'C'), ('B', 'C', 'D'), ('B', 'D', 'D'), ('C', 'C', 'C'), ('C', 'C', 'D'), ('C', 'D', 'D'), ('D', 'D', 'D'))
    >>> tcombinations_with_replacement('ABCD', 4)
    (('A', 'A', 'A', 'A'), ('A', 'A', 'A', 'B'), ('A', 'A', 'A', 'C'), ('A', 'A', 'A', 'D'), ('A', 'A', 'B', 'B'), ('A', 'A', 'B', 'C'), ('A', 'A', 'B', 'D'), ('A', 'A', 'C', 'C'), ('A', 'A', 'C', 'D'), ('A', 'A', 'D', 'D'), ('A', 'B', 'B', 'B'), ('A', 'B', 'B', 'C'), ('A', 'B', 'B', 'D'), ('A', 'B', 'C', 'C'), ('A', 'B', 'C', 'D'), ('A', 'B', 'D', 'D'), ('A', 'C', 'C', 'C'), ('A', 'C', 'C', 'D'), ('A', 'C', 'D', 'D'), ('A', 'D', 'D', 'D'), ('B', 'B', 'B', 'B'), ('B', 'B', 'B', 'C'), ('B', 'B', 'B', 'D'), ('B', 'B', 'C', 'C'), ('B', 'B', 'C', 'D'), ('B', 'B', 'D', 'D'), ('B', 'C', 'C', 'C'), ('B', 'C', 'C', 'D'), ('B', 'C', 'D', 'D'), ('B', 'D', 'D', 'D'), ('C', 'C', 'C', 'C'), ('C', 'C', 'C', 'D'), ('C', 'C', 'D', 'D'), ('C', 'D', 'D', 'D'), ('D', 'D', 'D', 'D'))
    >>> tcombinations_with_replacement('ABC', 4)
    (('A', 'A', 'A', 'A'), ('A', 'A', 'A', 'B'), ('A', 'A', 'A', 'C'), ('A', 'A', 'B', 'B'), ('A', 'A', 'B', 'C'), ('A', 'A', 'C', 'C'), ('A', 'B', 'B', 'B'), ('A', 'B', 'B', 'C'), ('A', 'B', 'C', 'C'), ('A', 'C', 'C', 'C'), ('B', 'B', 'B', 'B'), ('B', 'B', 'B', 'C'), ('B', 'B', 'C', 'C'), ('B', 'C', 'C', 'C'), ('C', 'C', 'C', 'C'))
    """
    return tuple(combinations_with_replacement(iterable, r))


def tcompact(iterable):
    """
    >>> tcompact((0, 1, 2, False, True, None))
    (1, 2, True)
    """
    return tuple(compact(iterable))


def treject(predicate, iterable):
    """
    >>> treject(lambda x: x > 1, [0, 1, 2])
    (0, 1)
    """
    return tuple(reject(predicate, iterable))


def tncycles(iterable, n):
    """
    >>> tncycles([1, 2, 3], 2)
    (1, 2, 3, 1, 2, 3)
    """
    return tuple(ncycles(iterable, n))


def trepeatfunc(func, times=None, *args):
    """
    >>> trepeatfunc(lambda : 1, times=3)
    (1, 1, 1)
    """
    return tuple(repeatfunc(func, times=times, *args))


def tgrouper(n, iterable, fillvalue=None):
    """
    >>> tgrouper(2, [1, 2, 3, 4, 5], 0)
    ((1, 2), (3, 4), (5, 0))
    """
    return tuple(grouper(n, iterable, fillvalue=fillvalue))


def troundrobin(*iterables):
    """
    >>> troundrobin([1, 2, 3], [4], [5, 6])
    (1, 4, 5, 2, 6, 3)
    """
    return tuple(roundrobin(*iterables))


def tsplitin(pred, iterable):
    """
    >>> tsplitin(lambda x: x % 2 == 0, [1, 2, 3, 4, 5])
    ((1, 3, 5), (2, 4))
    """
    return tuple(map(tuple, splitin(pred, iterable)))


def tsplitat(t, iterable):
    """
    >>> tsplitat(2, range(5))
    ((0, 1), (2, 3, 4))
    """
    return tuple(map(tuple, splitat(t, iterable)))


def tsplitby(pred, iterable):
    """
    >>> tsplitby(lambda x: x < 1, range(3))
    ((0,), (1, 2))
    """
    return tuple(map(tuple, splitby(pred, iterable)))


def tpowerset(iterable):
    """
    >>> tpowerset([1, 2, 3])
    ((), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3))
    """
    return tuple(powerset(iterable))


def tpairwise(iterable):
    """
    >>> tpairwise([1, 2, 3, 4])
    ((1, 2), (2, 3), (3, 4))
    """
    return tuple(pairwise(iterable))


def titer_except(func, exception, first_=None):
    """
    >>> d = {1: 1, 2: 2, 3: 3}
    >>> titer_except(d.popitem, KeyError)
    ((3, 3), (2, 2), (1, 1))
    """
    return tuple(iter_except(func, exception, first_=first_))


def tflatten(items):
    """
    >>> tflatten([1, (2, 3), (4, 5, (6, (7, (8,))))])
    (1, 2, 3, 4, 5, 6, 7, 8)
    """
    return tuple(flatten(items))
