#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Module containing functions for local search."""

import itertools

import split


def single_switch_local_search(initial_split, impurity_fn):
    """Does a local search, switching the side of one node at a time.

    impurity_fn receives left_values and right_values and return the impurity.
    """
    best_split = split.Split(left_values=initial_split.left_values,
                             right_values=initial_split.right_values,
                             impurity=initial_split.impurity)
    found_improvement = True
    while found_improvement:
        found_improvement = False
        for value in best_split.left_values:
            curr_left_values = best_split.left_values - set([value])
            curr_right_values = best_split.right_values + set([value])
            curr_split = split.Split(
                left_values=curr_left_values,
                right_values=curr_right_values,
                impurity=impurity_fn(curr_left_values, curr_right_values))
            if curr_split.is_better_than(best_split):
                found_improvement = True
                best_split = curr_split
                break
        if found_improvement:
            continue
        for value in best_split.right_values:
            curr_left_values = best_split.left_values + set([value])
            curr_right_values = best_split.right_values - set([value])
            curr_split = split.Split(
                left_values=curr_left_values,
                right_values=curr_right_values,
                impurity=impurity_fn(curr_left_values, curr_right_values))
            if curr_split.is_better_than(best_split):
                found_improvement = True
                best_split = curr_split
                break
    return best_split


def double_switch_local_search(initial_split, impurity_fn):
    """Does a local search, switching sides of two nodes at a time, one in each.

    impurity_fn receives left_values and right_values and return the impurity.
    """
    best_split = split.Split(left_values=initial_split.left_values,
                             right_values=initial_split.right_values,
                             impurity=initial_split.impurity)
    found_improvement = True
    while found_improvement:
        found_improvement = False
        for left_value, right_value in itertools.product(
                best_split.left_values, best_split.right_values):
            curr_left_values = (best_split.left_values -
                                set([left_value]) +
                                set([right_value]))
            curr_right_values = (best_split.right_values +
                                 set([left_value]) -
                                 set([right_value]))
            curr_split = split.Split(
                left_values=curr_left_values,
                right_values=curr_right_values,
                impurity=impurity_fn(curr_left_values, curr_right_values))
            if curr_split.is_better_than(best_split):
                found_improvement = True
                best_split = curr_split
                break
    return best_split
