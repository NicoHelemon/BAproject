# Contains path support functions.

import os

def index_last(L, e):
    return len(L) - 1 - L[::-1].index(e)

def goback_from_current_dir(i, dir = None):
    assert i >= 0

    if dir is None:
        dir = os.getcwd()

    while i > 0:
        dir = dir[:index_last(dir, '\\')]
        i -= 1

    return dir + '\\'