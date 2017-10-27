# -*- coding: utf-8 -*-
"""单元测试文件"""

from __future__ import absolute_import, division, print_function, unicode_literals  # NOQA

import sys
import unittest


def main():
    suite = unittest.defaultTestLoader.discover(
        '.', pattern='test_*.py', top_level_dir=None)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        sys.exit(1)


if __name__ == "__main__":
    main()
