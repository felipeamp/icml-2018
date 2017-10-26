#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Module containing the Split class and utility functions."""

import itertools
import math

import numpy as np

EPSILON = 1e-6


class Split(object):
    """Contains information about a node split and its impurity."""
    def __init__(self, left_values=None, right_values=None,
                 impurity=float('+inf')):
        if left_values is None:
            self.left_values = set()
        else:
            self.left_values = left_values
        if right_values is None:
            self.right_values = set()
        else:
            self.right_values = right_values
        self.impurity = impurity
        self.iteration_number = None
        self.superclasses_largest_frequence = None

    def set_iteration_number(self, iteration_number):
        """Saves iteration number of the given split."""
        self.iteration_number = iteration_number


    def set_superclasses_largest_frequence(self, largest_frequence):
        """Saves frequence of most frequent superclass."""
        self.superclasses_largest_frequence = largest_frequence

    def is_valid(self):
        """Indicates whether the Split is valid."""
        return ((self.left_values or self.right_values) and
                math.isfinite(self.impurity))

    def is_better_than(self, other_split):
        """Indicates whether the self Split is better than the given one."""
        return self.is_valid() and self.impurity < other_split.impurity

    def __eq__(self, other_split):
        """Indicates whether the self Split is equal to the given one, in values OR in impurity."""
        return ((self.left_values == other_split.left_values and
                 self.right_values == other_split.right_values) or
                math.isclose(self.impurity, other_split.impurity))


def powerset_using_symmetry(values):
    """Generates all non-empty possible splits for a set of values.

    Symmetric splits are considered equal. For instance, if values = {0, 1, 2},
    it will return [(0), (1), (2)]. If values = {0, 1, 2, 3}, it will return
    [(0), (1), (2), (3), (0, 1), (0, 2), (1, 2)].
    """
    if not values:
        return []
    elements = list(values)
    if len(elements) & 1:  # odd number of values
        return itertools.chain.from_iterable(
            itertools.combinations(elements, r)
            for r in range(1, (len(elements) // 2) + 1))
    uneven_splits = itertools.chain.from_iterable(
        itertools.combinations(elements, r)
        for r in range(1, (len(elements) // 2)))
    split_in_half = itertools.combinations(elements[:-1],
                                           len(elements) // 2)
    return itertools.chain.from_iterable([uneven_splits, split_in_half])


def get_num_samples_per_side(num_samples, num_samples_per_value, left_values,
                             right_values):
    """Returns two sets, each containing the values of a split side."""
    if len(left_values) <= len(right_values):
        num_left_samples = sum(num_samples_per_value[value]
                               for value in left_values)
        num_right_samples = num_samples - num_left_samples
    else:
        num_right_samples = sum(num_samples_per_value[value]
                                for value in right_values)
        num_left_samples = num_samples - num_right_samples
    return  num_left_samples, num_right_samples


def get_num_samples_per_class_in_values(contingency_table, values):
    """Returns a list, i-th entry contains the number of samples of class i."""
    num_classes = contingency_table.shape[1]
    num_samples_per_class = [0] * num_classes
    for value in values:
        for class_index in range(num_classes):
            num_samples_per_class[class_index] += contingency_table[
                value, class_index]
    return num_samples_per_class


def get_num_samples_per_class(contingency_table):
    """Returns a list, i-th entry contains the number of samples of class i."""
    num_values = contingency_table.shape[0]
    return get_num_samples_per_class_in_values(contingency_table,
                                               set(range(num_values)))


def get_smaller_set(set_1, set_2):
    """Returns the smaller of the sets (in length)."""
    if len(set_1) <= len(set_2):
        return set_1
    return set_2


def get_num_samples_per_value_in_classes(contingency_table, classes):
    """Returns a list, i-th entry contains the number of samples of value i."""
    return get_num_samples_per_class_in_values(np.copy(contingency_table).T, classes)


def get_num_samples_per_value_from_class(contingency_table, class_index):
    """Returns a list, i-th entry contains the number of samples of value i."""
    return get_num_samples_per_value_in_classes(contingency_table,
                                                set([class_index]))
