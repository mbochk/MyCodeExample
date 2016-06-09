from __future__ import print_function

import os
import sys
import shutil

from operator import add
from math import log

from tempfile import NamedTemporaryFile
from fileinput import input
from glob import glob

from pyspark import SparkContext


# Wrappers

# Context manager for auto-closing SparkContext
class SafeContext(object):
    def __init__(self, appName):
        self.appName = appName

    def __enter__(self):
        self.sc = SparkContext(appName=self.appName)
        return self.sc

    def __exit__(self, *kwargs):
        self.sc.stop()


class FancyFileSaver(object):
    def __init__(self, sc, output_path):
        self.sc = sc
        self.path = output_path
        self.task_count = 0

    def __call__(self, data, mapper):
        self.task_count += 1
        path = self.path + "_task{}".format(self.task_count)

        # Remove HDFS or from local storage
        if os.name == 'nt':
            try:
                shutil.rmtree(path)
            except:
                pass

        if isinstance(data, list):
            data = self.sc.parallelize(data)

        data.map(mapper).coalesce(1, shuffle=True) \
            .saveAsTextFile(path)
        print("Task{} Complete!".format(self.task_count))


def union(s1, s2):
    return s1.union(s2)


def union_add(s, val):
    return s.union(set([val]))


def value_sort(elem1, elem2):
    return

# data interpratation

def log10(count):
    if count < 2:
        return 0
    return int(log(count-1, 10))


def print2items(pair):
    return "{} {}".format(*pair)


def print_power_counts(counts):
    for i, pair in enumerate(counts):
        power = 10 ** pair[0]
        print("[{}, {}]\t{}".format(power, power * 10, pair[1]))


def print_counts(top_users):
    for user, count in top_users:
        print("{}\t{}\t{}".format(user, "username", count))


def printIntervalMap(pair):
    power = 10 ** int(pair[0])
    if power == 1:
        return "[{}, {}]\t{}".format(power, power*10, pair[1])
    else:
        return "({}, {}]\t{}".format(power, power*10, pair[1])


def count2strMap(pair):
    return "{}\t{}\t{}".format(pair[0], "username", pair[1])


def printCountNameMap(item):
    return "{}\t{}\t{}".format(item[0], item[1][1], item[1][0])


class FilterUserlist(object):
    def __init__(self, sc, user_list, position=0):
        self.list = sc.broadcast(user_list)
        self.position = position

    def __call__(self, pair):
        return pair[self.position] in self.list.value


# RddCount

def RddCount(rdd):
    return rdd.aggregateByKey(0, lambda x, y: x+1, add) \
        .sortBy(lambda (x, y): -y)

# Tasks


def task1(rdd):
    rdd_count = RddCount(rdd).cache()

    ratio = rdd.count() * 1.0 / rdd_count.count()
    print("Average follower count is {}".format(ratio))

    return rdd_count, ratio


def task2(rdd_count):
    interval_count = rdd_count \
        .map(lambda (user_id, follow_count): (log10(follow_count), 1)) \
        .reduceByKey(add) \
        .collect()

    interval_count.sort()
    return interval_count


def task3(rdd_count):
    top_num = 50
    top_users_count = rdd_count.takeOrdered(
        top_num, lambda (user_id, follower_count): -follower_count)

    filt = FilterUserlist(rdd_count.context,
                          [user for user, count in top_users_count])
    return filt, rdd_count.filter(filt)


def task4(rdd, top_filt):
    rdd_filter = rdd.filter(top_filt).cache()

    rdd_top_inverse = rdd_filter \
        .map(lambda (x, y): (y, x))

    rdd_top = rdd_top_inverse.join(rdd) \
        .map(lambda (x, y_z): y_z) \

    rdd_top = (rdd_top + rdd_filter).map(tuple).distinct()

    return RddCount(rdd_top)


def main():
    if len(sys.argv) != 4:
        print("Usage: topwords <input> <input2> <output>", file=sys.stderr)
        exit(-1)

    input_path = sys.argv[1]
    names_path = sys.argv[2]
    output_path = sys.argv[3]

    with SafeContext("Bochkarev_twitter") as sc:
        print(input_path)
        rdd = sc.textFile(input_path) \
            .map(lambda line: line.split('\t')) \
            .cache()
        print(rdd.count())

        output = FancyFileSaver(sc, output_path)

        rdd_names = sc.textFile(names_path) \
            .map(lambda line: line.split(' ')) \
            .cache()

        # task 1

        rdd_count, ratio = task1(rdd)
        output([ratio], str)

        # task 2

        interval_count = task2(rdd_count)
        output(interval_count, printIntervalMap)

        # task 3
        top_filt, rdd_top = task3(rdd_count)
        output(rdd_top.join(rdd_names), printCountNameMap)

        # task 4

        rdd_top_count = task4(rdd, top_filt)
        output(rdd_top_count.join(rdd_names), printCountNameMap)




if __name__ == "__main__":
    main()
