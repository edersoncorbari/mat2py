# -*- coding: utf-8 -*-
#
# @file precision_test.py
#
# Implementation of precision unit test class.
#
# MAT2PY - Portability of Matlab Functions for Python.
#
# Copyright (c) 2018-2020 Ederson Corbari
#
# @author Ederson Corbari
#
# $Id: EDMC Exp$
#

import sys
import os
import unittest
import time
import numpy
import pandas

from datetime import datetime
from precision import Precision


class TestPrecision(unittest.TestCase):

    def test_tic_toc(self):
        p = Precision()
        p.tic()
        time.sleep(2)
        p.toc()

    def test_datenum(self):
        p = Precision()
        d = [
            i for i in p.dtrange(
                datetime(
                    2018, 6, 12), datetime(
                    2025, 12, 12), {
                    'days': 1, 'hours': 2})]
        x = [p.datenum(i.date()) for i in d]
        self.assertEqual(len(x), 2530, 'Failed datenum!')
        # y = [p.datenum(i.year, i.month, i.day) for i in d]
        self.assertEqual(len(x), 2530, 'Failed datenum 2!')

    def test_prctile(self):
        p = Precision()
        d = [
            i for i in p.dtrange(
                datetime(
                    2018, 6, 12), datetime(
                    2059, 12, 12), {
                    'days': 1, 'hours': 2})]
        x = [p.datenum(i.date()) for i in d]
        self.assertEqual(len(x), 13992, 'Failed datenum!')

        x1 = p.prctile(x, 5)
        x2 = p.prctile(x, 95)
        r = (x2 - x1)

        self.assertEqual(x1, 737980.1, 'Failed prctile 5 low!')
        self.assertEqual(x2, 751621.9, 'Failed prctile 95 high!')
        self.assertEqual(r, 13641.800000000047, 'Failed prctile delta r!')

    def test_cell2mat(self):
        p = Precision()
        m = [[1, 2], [3, 4]]
        self.assertEqual(len(m), len(p.cell2mat(m)), 'Failed matrix!')
        self.assertEqual(2, len(p.cell2mat('1 2; 3 4')), 'Failed str!')

    def test_str2num(self):
        p = Precision()
        self.assertEqual(5, p.str2num('5'), 'Failed int to str!')
        self.assertEqual(5.2, p.str2num('5.2'), 'Failed float to str!')
        self.assertEqual(
            5.459999,
            p.str2num('5.459999'),
            'Failed float to str!')

    def test_num2str(self):
        p = Precision()
        self.assertEqual('5', p.num2str(5), 'Failed str to num!')
        self.assertEqual('5.2', p.num2str(5.2), 'Failed str to float!')
        self.assertEqual(
            '5.459999',
            p.num2str(5.459999),
            'Failed str to float!')

    def test_sprintf(self):
        p = Precision()
        self.assertEqual('50', p.sprintf('%d', 50), 'Failed sprintf %d!')
        self.assertEqual('WORK', p.sprintf('%s', 'WORK'), 'Failed sprintf %s!')

    def test_strcmp(self):
        p = Precision()
        self.assertEqual(True, p.strcmp('A', 'A'), 'Failed strcmp 0!')
        self.assertEqual(False, p.strcmp('B', 'C'), 'Failed strcmp 1!')

    def test_num2cell(self):
        p = Precision()
        x = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], numpy.int64)
        z = p.num2cell(x)
        self.assertEqual(3, len(z), 'Failed num2cell!')
        x = numpy.ones((170, 30))
        z = p.num2cell(x)
        self.assertEqual(170, len(z), 'Failed num2cell!')

    def test_strcat(self):
        p = Precision()
        df = pandas.DataFrame(
            data={
                'A': [
                    1, 2], 'B': [
                    3, 4]}, dtype=numpy.int8)
        self.assertEqual(list(['1', '2']), p.strcat(
            df, 'A'), 'Failed num2cell A!')
        self.assertEqual(list(['3', '4']), p.strcat(
            df, 'B'), 'Failed num2cell B!')

    def test_unique(self):
        p = Precision()
        l0 = [0, 1, 1, 2, 3, 4, 4, 5, 5, 6, 7, 7, 7]
        self.assertEqual(dict({0: 4, 1: 5, 2: 6, 3: 7, 4: 8, 5: 9, 6: 10, 7: 11}), p.unique(
            l0, 4), 'Failed unique 1!')
        self.assertEqual(list([0, 1, 2, 3, 4, 5, 6, 7]),
                         p.unique(l0), 'Failed unique 2!')

        x = p.unique(numpy.array([l0]))
        self.assertEqual([0, 1, 2, 3, 4, 5, 6, 7], list(
            x[0][0].flatten()), 'Failed unique 3!')
        self.assertEqual([0, 1, 3, 4, 5, 7, 9, 10], list(
            x[0][1].flatten()), 'Failed unique 4!')
        self.assertEqual([0, 1, 1, 2, 3, 4, 4, 5, 5, 6, 7, 7, 7],
                         list(x[0][2].flatten()), 'Failed unique 5!')
        self.assertEqual([1, 2, 1, 1, 2, 2, 1, 3], list(
            x[0][3].flatten()), 'Failed unique 6!')

    def test_histc(self):
        p = Precision()
        v = numpy.array([[1.5, 2.0, 3], [4, 5.9, 6]], numpy.int64)
        x = p.histc(v, numpy.amax(v) + 1)
        self.assertEqual([1, 1, 1, 0, 1, 1, 1], list(
            x[0].flatten()), 'Failed histc 1!')
        self.assertEqual([1.0,
                          1.7142857142857144,
                          2.428571428571429,
                          3.142857142857143,
                          3.857142857142857,
                          4.571428571428571,
                          5.285714285714286,
                          6.0],
                         list(x[1].flatten()),
                         'Failed histc 1!')

    def test_zero_or_one(self):
        p = Precision()
        l0 = ['A', 'B', 'A', 'B', 'C']
        self.assertEqual([1, 0, 1, 0, 0], p.zero_or_one(
            l0, 'A'), 'Failed zero or one A!')
        self.assertEqual([0, 1, 0, 1, 0], p.zero_or_one(
            l0, 'B'), 'Failed zero or one B!')
        self.assertEqual([0, 0, 0, 0, 1], p.zero_or_one(
            l0, 'C'), 'Failed zero or one C!')

    def test_overlap1d(self):
        p = Precision()
        l0 = ['A', 'B', 'A']
        l1 = ['A', 'D', 'A']
        self.assertEqual([[0, ('A', 'A')], [2, ('A', 'A')]],
                         p.overlap1d(l0, l1), 'Failed overlap1d 1!')
        self.assertEqual([[0, 2], [], [0, 2]], list(
            p.overlap1d(l0, l1, 0)), 'Failed overlap1d 2!')

    def test_overlap2d(self):
        p = Precision()
        a, b = p.overlap2d(numpy.array(
            [1, 2, 4, 5]), numpy.array([4, 6, 10, 9, 1]))
        self.assertEqual([0, 2], list(a.flatten()), 'Failed overlap2d 1!')
        self.assertEqual([4, 0], list(b.flatten()), 'Failed overlap2d 2!')


if __name__ == '__main__':
    unittest.main()
