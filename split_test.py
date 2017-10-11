#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Tests code in the split module."""

from itertools import chain, combinations
import unittest

import split


class TestSplit(unittest.TestCase):

    def test_iteration_number(self):
        default_split = split.Split()
        default_split.set_iteration_number(10)

        self.assertEqual(10, default_split.iteration_number)

    def test_default_is_not_valid(self):
        default_split = split.Split()

        self.assertFalse(default_split.is_valid())

    def test_is_valid(self):
        curr_split = split.Split(left_values=set([1]), right_values=set([2]), impurity=1.0)

        self.assertTrue(curr_split.is_valid())

    def test_is_better_than(self):
        default_split_1 = split.Split()
        default_split_2 = split.Split()
        split_1 = split.Split(left_values=set([1]), right_values=set([2, 3]), impurity=1.0)
        split_2 = split.Split(left_values=set([1, 2]), right_values=set([3]), impurity=2.0)

        self.assertTrue(split_1.is_better_than(split_2))
        self.assertFalse(split_2.is_better_than(split_1))
        self.assertTrue(split_1.is_better_than(default_split_1))
        self.assertTrue(split_2.is_better_than(default_split_1))
        self.assertFalse(default_split_1.is_better_than(default_split_2))
        self.assertFalse(default_split_2.is_better_than(default_split_1))

    def test_eq(self):
        default_split = split.Split()
        split_1 = split.Split(left_values=set([1]), right_values=set([2]), impurity=1.0)
        split_2 = split.Split(left_values=set([1]), right_values=set([2]), impurity=2.0)
        split_3 = split.Split(left_values=set([1, 2]), right_values=set([2]), impurity=2.0)
        split_4 = split.Split(left_values=set([1]), right_values=set([2, 3]), impurity=2.0)

        self.assertFalse(split_1 == default_split)
        self.assertTrue(split_1 == split_2)
        self.assertFalse(split_1 == split_3)
        self.assertFalse(split_1 == split_4)
        self.assertTrue(split_2 == split_3)
        self.assertTrue(split_2 == split_4)


if __name__ == '__main__':
    unittest.main()