# -*- coding: utf-8 -*
""" 补充扩展 fn 和 toolz 中没有的迭代工具库 """

from __future__ import absolute_import, division, print_function, unicode_literals  # NOQA

from functools import partial
from fn import F
from fn.monad import Empty
from fn.iters import compact, first
from toolz.compatibility import filter

from .func import is_not_none, is_option_full

__all__ = ("every", "some", "first_true", "first_object", "first_option_full",
           "first_pred_object")


every = F(partial(map)) >> all
some = F(partial(map)) >> compact >> first


# 从序列中选择第一个为 真 的元素，可以设置找不到的默认值和谓词
# [None, 0, {}, 1] -> 1
def first_true(iterable, default=False, pred=None):
    return next(filter(pred, iterable), default)


# 从序列中选择第一个不为 None 的元素
# [None, 0, {}] -> 0
def first_object(iterable, pred=is_not_none):
    return first_true(iterable, default=None, pred=pred)


# 从序列中选择第一个为 Full 的元素，找不到默认为 Empty
# [Empty(), Full(0)] -> Full(0)
def first_option_full(iterable, pred=is_option_full):
    return first_true(iterable, default=Empty, pred=pred)


# 从序列中选择第一个被 pred 的执行过但不返回 None 的这个值
# 仅应当用于 pred 结果用 None 表示的语义
# [1, 2, 3]
# pred=lambda x: "str object"
# -> "str_object"  并且 2 以后不会被遍历
# pred=lambda x: None
# -> None 会遍历所有迭代
def first_pred_object(iterable, pred=is_not_none):
    for i in iterable:
        obj = pred(i)
        if obj is not None:
            return obj
    return None
