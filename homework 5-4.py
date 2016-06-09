# -*- coding: utf-8 -*-
"""
Created on Thu May 19 20:09:18 2016

@author: mbochk
"""

import sys
from contextlib import contextmanager


@contextmanager
def AssertRaises(exc):
    try:
        yield
    except exc:
        return
    except Exception:
        raise AssertionError
    raise AssertionError


exec(sys.stdin.read())
