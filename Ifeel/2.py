# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 18:15:08 2016

@author: mbochk
"""


def calculate_devide_indexes(N, M):
    left_residue = (N % M) / 2
    part_size = N / M
    return [[i, i + part_size - 1] for i in xrange(left_residue,
            N - part_size + 1, part_size)]


def print_devide_indexes(indexes):
    for i in indexes:
        print i


def main():
    import sys
    N, M = sys.argv[1], sys.argv[2]
    N, M = int(N), int(M)

    indexes = calculate_devide_indexes(N, M)
    print_devide_indexes(indexes)


if __name__ == "__main__":
    main()
