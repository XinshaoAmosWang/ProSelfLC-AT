# -*- coding: utf-8 -*-

import pytest

from proselflc.skeleton import fib

__author__ = "XinshaoAmosWang"
__copyright__ = "XinshaoAmosWang"
__license__ = "gpl3"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
