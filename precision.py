# -*- coding: utf-8 -*-
#
# @file precision.py
#
# Implementation of precision class.
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
import pandas
import numpy
import datetime
import time
import itertools

from functools import reduce
from multipledispatch import dispatch


class Precision(object):
    """
    This class is used for calculations and also aims to simulate
    some functions and behaviors of MATLAB.

    :Example: :func:`prctile`, :func:`cell2mat`, :func:`datenum`.
    """

    def __init__(self):
        """
        Initialization of the class. Standard loading variables:

        - :obj:`m_info`: Information data used by the logging engine.
        - :obj:`__m_err`: Default error message for asserts.
        - :obj:`__m_start_time`: Used to calculate the elapsed time.
        """
        self.m_info = self.__class__.__module__ + '.' + self.__class__.__name__
        self.__m_err = ('Ops... The expected value hede is {}!')
        self.__m_tic_start = time.time()

    def tic(self) -> None:
        """
        Tic starts a stopwatch timer to measure performance. The function
        records the internal time at execution of the tic functions. Display the
        elapsed time with the :meth:`toc` method.

        .. seealso::
            - http://www.mathworks.com/help/ref/tic.html
        """
        self.__m_tic_start = time.time()

    def toc(self) -> None:
        """
        Toc reads the elapsed time from the stopwatch timer started by
        the :meth:`tic` method. The function reads the internal time at the execution
        of the toc method, and displays the elapsed time since the most recent
        call to the tic function that had no output, in seconds.

        .. seealso::
            - http://www.mathworks.com/help/ref/toc.html
        """
        end = time.time()
        tmp = (end - self.__m_tic_start)
        hours = (tmp // 3600)
        tmp = (tmp - 3600 * hours)
        minutes = (tmp // 60)
        seconds = (tmp - 60 * minutes)
        s = 'secounds' if seconds > 0 else s
        s = 'minutes' if minutes > 0 else s
        s = 'hours' if hours > 0 else s
        m = (': > Elapsed time is %d:%d:%d {}.' % (
            hours, minutes, seconds)).format(s)
        print(self.sprintf("%s\n", m))

    def dtrange(
            self,
            start: datetime.datetime,
            end: datetime.datetime,
            delta: dict) -> datetime.datetime:
        """
        Creates a range of dates with start and end.

        :param start: A start date.
        :type start: datetime.datetime
        :param end: A end date.
        :type end: datetime.datetime
        :param delta: A dictionary with day and time.
        :type delta: dict
        :returns: A list with date range.
        :rtype: datetime.datetime.

        Raises:
            - :obj:`Exception` Some other error not expected.
        """
        try:
            current = start
            if not isinstance(delta, datetime.timedelta):
                delta = datetime.timedelta(**delta)
            while current < end:
                yield current
                current += delta
        except Exception as e:
            print(self.m_info, ': Exception = ', str(e))

    def quantile(self, x: list, q: numpy.float64) -> float:
        """
        Returns quantiles of the values in data matrix.

        :param x: List with the combination of numbers.
        :param q: Size numers.
        :type x: list
        :type q: float64
        :returns: The cumulative probability or probabilities.
        :rtype: float

        .. note::
            There may be slight variations in results compared to MATLAB.

        .. seealso::
            - http://www.mathworks.com/help/stats/quantile.html

        Raises:
            - :obj:`Exception` Some other error not expected.
        """
        assert len(x) >= 1, self.__m_err.format('>=1')
        assert q >= 0.0, self.__m_err.format('>=1')
        try:
            n = len(x)
            y = numpy.sort(x)
            return float((numpy.interp(q, numpy.linspace(
                1 / (2 * n), (2 * n - 1) / (2 * n), n), y)))
        except Exception as e:
            print(self.m_info, ': Exception = ', str(e))
        return float(0.0)

    def prctile(self, x: list, p: int) -> float:
        """
        Returns percentiles of the values in a data matrix or array
        for the percentages X in the interval [0,100]. This function
        depenends on :func:`quantile`.

        :param x: List with the combination of numbers.
        :param q: Size numers.
        :type x: list
        :type q: int
        :returns: The percentages in a range of [0,100].
        :rtype: float

        .. note::
            There may be slight variations in results compared to MATLAB.

        .. seealso::
            - http://www.mathworks.com/help/stats/prctile.html

        Raises:
            - :obj:`Exception` Some other error not expected.
        """
        assert len(x) >= 1, self.__m_err.format('>=1')
        assert p >= 1, self.__m_err.format('>=1')
        try:
            return (self.quantile(x, numpy.array(p) / 100))
        except Exception as e:
            print(self.m_info, ': Exception = ', str(e))
        return float(0.0)

    @dispatch(datetime.date)
    def datenum(self, d: datetime.date) -> int:
        """
        Converts a date or time to serial date or time.

        :param d: An ISO date.
        :type d: date
        :returns: Return the proleptic Gregorian oridinal dates.
        :rtype: int

        .. seealso::
            - https://www.mathworks.com/help/matlab/ref/datenum.html

        Raises:
            - :obj:`Exception` Some other error not expected.
        """
        try:
            return datetime.date.toordinal(
                datetime.date(d.year, d.month, d.day)) + 366
        except Exception as e:
            print(self.m_info, ': Exception = ', str(e))
        return int(0)

    @dispatch(pandas._libs.tslib.Timestamp)
    def datenum(self, d: pandas._libs.tslib.Timestamp) -> int:
        """
        Converts a date or time to serial date or time.

        :param d: An ISO date.
        :type d: pandas._libs.tslib.Timestamp
        :returns: Return the proleptic Gregorian oridinal dates.
        :rtype: int

        Raises:
            - :obj:`Exception` Some other error not expected.
        """
        try:
            return self.datenum(datetime.date(d.year, d.month, d.day))
        except Exception as e:
            print(self.m_info, ': Exception = ', str(e))
        return int(0)

    @dispatch(int, int, int)
    def datenum(self, y: int, m: int, d: int) -> int:
        """
        Converts a date or time to serial date or time.

        :param y: Year.
        :param m: Month.
        :param d: Day.
        :type y: int
        :type m: int
        :type d: int
        :returns: Return the proleptic Gregorian oridinal dates.
        :rtype: int

        Raises:
            - :obj:`Exception` Some other error not expected.
        """
        try:
            return self.datenum(datetime.date(y, m, d))
        except Exception as e:
            print(self.m_info, ': Exception = ', str(e))
        return int(0)

    @dispatch(list)
    def cell2mat(self, x: list) -> numpy.matrixlib.defmatrix.matrix:
        """
        Convert cell array of matrices to single matrix.

        :param x: A list.
        :type x: list
        :returns: Returns cell arrays.
        :rtype: numpy.matrix

        .. seealso::
            - https://www.mathworks.com/help/matlab/ref/cell2mat.html

        Raises:
            - :obj:`Exception` Some other error not expected.
        """
        try:
            return numpy.matrix(x)
        except Exception as e:
            print(self.m_info, ': Exception = ', str(e))

    @dispatch(str)
    def cell2mat(self, x: str) -> numpy.matrixlib.defmatrix.matrix:
        """
        Convert cell array of matrices to single matrix.

        :param x: A string separated by commas, e.g: "1 2; 3 4"
        :type x: str
        :returns: Returns cell arrays.
        :rtype: numpy.matrix

        Raises:
            - :obj:`Exception` Some other error not expected.
        """
        try:
            return numpy.matrix(x)
        except Exception as e:
            print(self.m_info, ': Exception = ', str(e))

    def str2num(self, x):
        """
        Convert string to number.

        :param x: A number of type float or int.
        :type x: float or int
        :returns: Returns the converted number.
        :rtype: float or int

        .. seealso::
            - https://www.mathworks.com/help/matlab/ref/str2num.html

        Raises:
            - :obj:`BaseException` A float/int conversion error may occur, we let pass.
        """
        assert len(x) >= 1, self.__m_err.format('>=1')
        r = list([])
        try:
            r = int(x)
        except BaseException:
            try:
                r = float(x)
            except BaseException:
                pass
        return r

    def num2cell(self, x: numpy.ndarray) -> numpy.ndarray:
        """
        Convert array to cell array with consistently sized cells.

        :param x: An array 2d.
        :type x: numpy.array
        :returns: Returns the converted cell array.
        :rtype: numpy.ndarray

        .. seealso::
            - https://www.mathworks.com/help/matlab/ref/num2cell.html

        Raises:
            - :obj:`Exception` Some other error not expected.
        """
        assert x.size >= 1, self.__m_err.format('>=1')
        try:
            if isinstance(x, numpy.ndarray):
                return [self.num2cell(i) for i in x]
        except Exception as e:
            print(self.m_info, ': Exception = ', str(e))
        return x

    def num2str(self, x: str) -> str:
        """
        Convert number to string.

        :param x: A string number.
        :type x: str
        :returns: Returns the converted string.
        :rtype: str

        .. seealso::
            - https://www.mathworks.com/help/matlab/ref/num2str.html

        Raises:
            - :obj:`Exception` Some other error not expected.
        """
        assert (len(str(x))) >= 1, self.__m_err.format('>=1')
        try:
            return str(x)
        except Exception as e:
            print(self.m_info, ': Exception = ', str(e))

    def sprintf(self, x, *args):
        """
        Write formated data to string.

        :param x: A string value.
        :type x:  str
        :args: Variable length argument list
        :type args: list
        :returns: Returns formated data to string.
        :rtype: str % args

        .. seealso::
            - https://www.mathworks.com/help/matlab/ref/sprintf.html

        Raises:
            - :obj:`Exception` Some other error not expected.
        """
        assert (len(str(x))) >= 1, self.__m_err.format('>=1')
        try:
            return (x % args)
        except Exception as e:
            print(self.m_info, ': Exception = ', str(e))

    def strcmp(self, a: str, b: str) -> bool:
        """
        Compare strings.

        :param a: A string.
        :param b: A string.
        :type a: str
        :type b: str
        :returns: Returns True if the string is equal otherwise False.
        :rtype: bool

        .. seealso::
            - https://www.mathworks.com/help/matlab/ref/strcmp.html

        Raises:
            - :obj:`Exception` Some other error not expected.
        """
        assert len(a) >= 1, self.__m_err.format('>=1')
        assert len(b) >= 1, self.__m_err.format('>=1')
        try:
            return a == b
        except Exception as e:
            print(self.m_info, ': Exception = ', str(e))

    def strcat(self, df: pandas.core.frame.DataFrame, k: str) -> list:
        """
        Concatenate strings horizontally.

        :param df: The pandas DataFrame.
        :param k: The column name to look up in the decision base.
        :returns: Returns a concatenated list.
        :rtype: list

        .. seealso::
            - https://www.mathworks.com/help/matlab/ref/strcat.html

        Raises:
            - :obj:`Exception` Some other error not expected.
        """
        assert df.size >= 1, self.__m_err.format('>=1')
        l = list([])
        try:
            for _i, r in df.iterrows():
                l.append(str(r[k]).upper())
        except Exception as e:
            print(self.m_info, ': Exception = ', str(e))
        return list(l)

    @dispatch(list, int)
    def unique(self, l: list, n: int) -> dict:
        """
        Find duplicate elements from a list by doing a sort and
        maintaining the position (index) of the element.

        :param l: An array 1d.
        :param n: Start from number count (X), best leave 1.
        :type l: list
        :type n: int
        :returns: Returns a dictionary sorted with unique elements and index position.
        :rtype: dict

        .. seealso::
            - https://www.mathworks.com/help/matlab/ref/unique.html

        Raises:
            - :obj:`Exception` Some other error not expected.
        """
        assert len(l) >= 1, self.__m_err.format('>=1')
        assert n >= 0, self.__m_err.format('>=0')
        try:
            return dict(zip(sorted(set(l)), itertools.count(n)))
        except Exception as e:
            print(self.m_info, ': Exception = ', str(e))

    @dispatch(list)
    def unique(self, l: list) -> list:
        """
        Find duplicate elements by maintaining order.

        :param l: An array 1d.
        :type l: list
        :returns: Returns a list of unique elements.
        :rtype: list

        .. seealso::
            - https://www.mathworks.com/help/matlab/ref/unique.html

        Raises:
            - :obj:`Exception` Some other error not expected.
        """
        assert len(l) >= 1, self.__m_err.format('>=1')
        try:
            return reduce(lambda y, x: y.append(
                x) or y if x not in y else y, l, [])
        except Exception as e:
            print(self.m_info, ': Exception = ', str(e))

    @dispatch(numpy.ndarray)
    def unique(self, l: numpy.ndarray) -> numpy.ndarray:
        """
        Find the unique elements of an array. It takes the duplicate items
        with the position, makes a count of how many times they appear (bincount),
        and the indexes of the unique matrix.

        :param l: An array 1d.
        :type l: numpy.ndarray
        :returns: Returns stack arrays in sequence vertically.
        :rtype: numpy.ndarray

        Take into consideration:
        - a = The unique values found.
        - b = The index position where they were found.
        - c = Number of times each item appears (bincount).
        - d = The index of each unique matrix.

        .. seealso::
            - https://www.mathworks.com/help/matlab/ref/unique.html
            - https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.unique.html

        Raises:
            - :obj:`Exception` Some other error not expected.
        """
        assert l.size >= 1, self.__m_err.format('>=1')
        try:
            a, b, c, d = numpy.unique(
                l, return_index=True, return_counts=True, return_inverse=True)
        except Exception as e:
            print(self.m_info, ': Exception = ', str(e))
        if a.size > 1:
            return numpy.vstack((numpy.array([[numpy.array(a)], [numpy.array([b], dtype=int)], [
                                numpy.array(c, dtype=int)], [numpy.array(d, dtype=int)]]))).T
        return numpy.vstack((numpy.array([[numpy.array(a)], [numpy.array([0], dtype=int)], [
                            numpy.array(0, dtype=int)], [numpy.array(0, dtype=int)]]))).T

    def histc(self, l: numpy.ndarray, b: numpy.int64) -> tuple:
        """
        Histogram bin counts.

        :param l: An array 2d.
        :param b: Numbers bincount.
        :returns: Returns a dictionary with value and position.
        :rtype: tuple

        .. seealso::
            - https://www.mathworks.com/help/matlab/ref/histc.html
            - https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html

        Raises:
            - :obj:`Exception` Some other error not expected.
        """
        assert l.size >= 1, self.__m_err.format('>=1')
        assert b >= 0, self.__m_err.format('>=1')
        try:
            return numpy.histogram(l, b)
        except Exception as e:
            print(self.m_info, ': Exception = ', str(e))

    @dispatch(list, list)
    def overlap1d(self, v0: list, v1: list) -> list:
        """
        It overlaps the vectors, considering the values that coincide, when
        this happens we saved the position that made it and make the return.

        :param v0: An array 1d.
        :param v1: An array 1d.
        :returns: Returns an index list.
        :rtype: list

        Raises:
            - :obj:`Exception` Some other error not expected.

        """
        assert len(v0) >= 1, self.__m_err.format('>=1')
        assert len(v1) >= 1, self.__m_err.format('>=1')
        assert len(v0) == len(v1), self.__m_err.format('==1')
        r = list([])
        try:
            for i, j in enumerate(zip(v0, v1)):
                if v0[i] == v1[i]:
                    r.append([i, j])
        except Exception as e:
            print(self.m_info, ': Exception = ', str(e))

        return r

    @dispatch(list, list, int)
    def overlap1d(self, v0: list, v1: list, n: int) -> object:
        """
        It makes an overlap by looking for the intersections with two
        vectors in normal array (list).

        :param v0: An array 1d.
        :param v1: An array 1d.
        :param n: Used not to confuse the dispatch.
        :type v0: list
        :type v1: list
        :type n: int
        :returns: Returns an index list.
        :rtype: list

        Raises:
            - :obj:`Exception` Some other error not expected.

        """
        assert len(v0) >= 1, self.__m_err.format('>=1')
        assert len(v1) >= 1, self.__m_err.format('>=1')
        assert len(v0) == len(v1), self.__m_err.format('==1')
        try:
            for i in v0:
                yield [p for p, j in enumerate(v1) if i == j]
        except Exception as e:
            print(self.m_info, ': Exception = ', str(e))

    def overlap2d(self, v0: numpy.ndarray, v1: numpy.ndarray) -> numpy.ndarray:
        """
        It makes an overlap by looking for the intersections with two
        numpy array, find index of the intersection.

        :param v0: An array 1d.
        :param v1: An array 1d.
        :type v0: numpy.ndarray
        :type v1: numpy.ndarray
        :returns: Returns an index list.
        :rtype: list

        .. seealso::
            - https://www.followthesheep.com/?p=1366
        """
        try:
            a1 = numpy.argsort(v0)
            b1 = numpy.argsort(v1)

            sort_left_a = v0[a1].searchsorted(v1[b1], side='left')
            sort_right_a = v0[a1].searchsorted(v1[b1], side='right')

            sort_left_b = v1[b1].searchsorted(v0[a1], side='left')
            sort_right_b = v1[b1].searchsorted(v0[a1], side='right')

            inds_b = (sort_right_a - sort_left_a > 0).nonzero()[0]
            inds_a = (sort_right_b - sort_left_b > 0).nonzero()[0]

            return (a1[inds_a], b1[inds_b])
        except Exception as e:
            print(self.m_info, ': Exception = ', str(e))

    def zero_or_one(self, l0: list, s: str) -> list:
        """
        Fill with zero or one, following the logic where I find the keyword
        add 1 or is zero.

        :param l0: Amount to be excluded (C).
        :param s: Amount to be exclused (R,E).
        :type l0: list
        :type s: str
        :returns: Returns a list with zero or one.
        :rtype: list

        Raises:
            - :obj:`Exception` Some other error not expected.

        """
        r0 = list([])
        try:
            for i in list(l0):
                r0.append(1 if i == s else 0)
        except Exception as e:
            print(self.m_info, ': Exception = ', str(e))
        return list(r0)
