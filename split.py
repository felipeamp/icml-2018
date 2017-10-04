#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Module containing the Split class and utility functions."""

import math

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

    def is_valid(self):
        """Indicates whether the Split is valid."""
        return (self.left_values is not None and
                self.right_values is not None and
                math.isfinite(self.impurity))

    def is_better_than(self, other_split):
        """Indicates whether the self Split is better than the given one."""
        return self.is_valid() and self.impurity < other_split.impurity


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
        num_left_samples = num_samples - right_values
    return  num_left_samples, num_right_samples


def get_num_samples_per_class(contingency_table, values):
    """Returns a list, i-th entry contains the number of samples of class i."""
    num_classes = contingency_table.shape[1]
    num_samples_per_class = [0] * num_classes
    for value in values:
        for class_index in range(num_classes):
            num_samples_per_class[class_index] += contingency_table[
                value, class_index]
    return num_samples_per_class
