# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 16:33:03 2016

@author: mbochk
"""

import sys


def calculate_prime_decomposition(N):
    """returns list of Counters
    for n = p1**n1 * p2**n2 * ...
    dcmp[n] = Counter({p1:n1, p2:n2})
    """

    from collections import Counter, defaultdict

    N += 1
    dcmp = defaultdict(Counter)

    for p in xrange(2, N):
        # check if p is prime
        if len(dcmp[p]) == 0:
            p_power = p

            # for each power of p increase degree of p in dcmp
            while p_power < N:
                for n in xrange(p_power, N, p_power):
                    dcmp[n][p] += 1
                p_power *= p
    return dcmp


def print_prime_decomposition(dcmp):
    for i, i_dcmp in enumerate(dcmp):
        print "{} = 1".format(i),
        for p, deg in i_dcmp.iteritems():
            print " * {}**{}".format(p, deg),
        print


def main():
    N = sys.argv[1]
    N = int(N)
    dcmp = calculate_prime_decomposition(N)
    print_prime_decomposition(dcmp)

if __name__ == "__main__":
    main()
