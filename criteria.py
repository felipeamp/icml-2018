#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Module containing all criteria available for tests."""

import math
import operator
import random

import numpy as np

# import local_search
import split

MAX_ITERATIONS = 100


class Criterion(object):
    """Finds the best split for all atributes using find_best_split_fn in each.
    """
    def __init__(self, name, find_best_split_fn):
        self.name = name
        self.find_best_split_fn = find_best_split_fn

    def find_all_best_splits(self, tree_node):
        """Returns a list with the best split for each attribute."""
        best_splits = []
        for (attrib_index, is_valid_nominal_attrib) in enumerate(
                tree_node.valid_nominal_attribute):
            if is_valid_nominal_attrib:
                attrib_best_split = self.find_best_split_fn(
                    tree_node, attrib_index)
                best_splits.append(attrib_best_split)
            else:
                best_splits.append([])
        return best_splits


def get_indices_count_sorted(num_samples_per_index):
    """Returns list of indices ordered by their count in num_samples_per_value.
    """
    num_samples_per_index_enumerated = list(
        enumerate(num_samples_per_index))
    num_samples_per_index_enumerated.sort(key=lambda x: x[1])
    return num_samples_per_index_enumerated


def get_indices_sorted_per_count(num_samples_per_index):
    """Returns list of indices ordered by their count in num_samples_per_value.
    """
    indices_count_sorted = get_indices_count_sorted(num_samples_per_index)
    return [index for (index, _) in indices_count_sorted]


def calculate_split_gini_index(num_samples, contingency_table,
                               num_samples_per_value, left_values,
                               right_values):
    """Calculates the weigthed Gini index of a split."""
    num_left_samples, num_right_samples = split.get_num_samples_per_side(
        num_samples, num_samples_per_value, left_values, right_values)
    num_samples_per_class_left = split.get_num_samples_per_class_in_values(
        contingency_table, left_values)
    left_gini = calculate_node_gini_index(num_left_samples,
                                          num_samples_per_class_left)
    num_samples_per_class_right = split.get_num_samples_per_class_in_values(
        contingency_table, right_values)
    right_gini = calculate_node_gini_index(num_right_samples,
                                           num_samples_per_class_right)
    return ((num_left_samples / num_samples) * left_gini +
            (num_right_samples / num_samples) * right_gini)


def calculate_node_gini_index(num_split_samples,
                              num_samples_per_class_in_split):
    """Calculates the Gini index of a node."""
    gini_index = 1.0
    for curr_class_num_samples in num_samples_per_class_in_split:
        gini_index -= (curr_class_num_samples / num_split_samples)**2
    return gini_index


def get_best_split(num_samples, contingency_table, num_samples_per_value,
                   node_impurity_fn):
    """Gets the best split using the two-class trick.

    Assumes contingency_table has only 2 columns.
    """
    def update_num_samples_per_class(contingency_table, value_switched_left,
                                     num_samples_per_class_left,
                                     num_samples_per_class_right):
        """Updates num_samples_per_class lists when switching value to left."""
        num_classes = contingency_table.shape[1]
        for class_index in range(num_classes):
            num_samples_per_class_left[class_index] += contingency_table[
                value_switched_left, class_index]
            num_samples_per_class_right[class_index] -= contingency_table[
                value_switched_left, class_index]

    assert contingency_table.shape[1] == 2
    num_values = len(num_samples_per_value)
    num_classes = contingency_table.shape[1]
    values_sorted_per_count = get_indices_sorted_per_count(
        contingency_table[:, 0])
    # We start with the (invalid) split where every value is on the right side.
    curr_split = split.Split(left_values=set(),
                             right_values=set(range(num_values)))
    best_split = curr_split
    # We use the four variables below to use dynamic programming on them. They
    # are needed to calculate the impurity of a split.
    num_left_samples = 0
    num_right_samples = num_samples
    num_samples_per_class_left = [0] * num_classes
    num_samples_per_class_right = split.get_num_samples_per_class_in_values(
        contingency_table, curr_split.right_values)
    for last_left_value in values_sorted_per_count[:-1]:
        left_values = curr_split.left_values + set([last_left_value])
        right_values = curr_split.right_values - set([last_left_value])
        # Update the variables needed for the impurity calculation using a
        # dynamic programming approach.
        num_left_samples += num_samples_per_value[last_left_value]
        num_right_samples -= num_samples_per_value[last_left_value]
        update_num_samples_per_class(contingency_table, last_left_value,
                                     num_samples_per_class_left,
                                     num_samples_per_class_right)
        # Impurity calculation for the split.
        left_impurity = node_impurity_fn(num_left_samples,
                                         num_samples_per_class_left)
        right_impurity = node_impurity_fn(num_right_samples,
                                          num_samples_per_class_right)
        split_impurity = (
            (num_left_samples / num_samples) * left_impurity +
            (num_right_samples / num_samples) * right_impurity)
        curr_split = split.Split(
            left_values=left_values,
            right_values=right_values,
            impurity=split_impurity)
        if curr_split.is_better_than(best_split):
            best_split = curr_split
    return best_split


def gini_gain(tree_node, attrib_index):
    """Gets the attribute's best split according to the Gini Gain.

    Calculates and returns the weighted Gini Index instead of the actual Gini
    Gain.
    """
    all_values = set(range(tree_node.contingency_tables[attrib_index].shape[0]))
    num_samples = tree_node.dataset.num_samples
    contingency_table = tree_node.contingency_tables[
        attrib_index].contingency_table
    num_samples_per_value = tree_node.contingency_tables[
        attrib_index].num_samples_per_value
    if tree_node.dataset.num_classes == 2:
        best_split = get_best_split(
            num_samples, contingency_table, num_samples_per_value,
            calculate_node_gini_index)
    else:
        best_split = split.Split()
        for left_values in split.powerset_using_symmetry(all_values):
            right_values = all_values - left_values
            split_gini_index = calculate_split_gini_index(num_samples,
                                                          contingency_table,
                                                          num_samples_per_value,
                                                          left_values,
                                                          right_values)
            curr_split = split.Split(left_values=left_values,
                                     right_values=right_values,
                                     impurity=split_gini_index)
            if curr_split.is_better_than(best_split):
                best_split = curr_split
    return best_split


def get_contingency_table_for_superclasses(
        num_values, contingency_table, num_samples_per_value, left_classes):
    """Sums the columns of the contingency table for left and right classes."""
    num_samples_per_value_from_left_classes = (
        split.get_num_samples_per_value_in_classes(contingency_table,
                                                   left_classes))
    num_samples_per_value_from_right_classes = [
        (num_samples_per_value[value] -
         num_samples_per_value_from_left_classes[value])
        for value in range(num_values)]
    superclasses_contingency_table = np.array(
        [num_samples_per_value_from_left_classes,
         num_samples_per_value_from_right_classes], dtype=int)
    return superclasses_contingency_table


def twoing(tree_node, attrib_index):
    """Gets the attribute's best split according to the Twoing criterion."""
    num_values, num_classes = tree_node.contingency_tables[attrib_index].shape
    all_classes = set(range(num_classes))
    num_samples = tree_node.dataset.num_samples
    contingency_table = tree_node.contingency_tables[
        attrib_index].contingency_table
    num_samples_per_value = tree_node.contingency_tables[
        attrib_index].num_samples_per_value
    best_split = split.Split()
    for left_classes in split.powerset_using_symmetry(all_classes):
        superclasses_contingency_table = get_contingency_table_for_superclasses(
            num_values, contingency_table, num_samples_per_value, left_classes)
        curr_split = get_best_split(num_samples,
                                    superclasses_contingency_table,
                                    num_samples_per_value,
                                    calculate_node_gini_index)
        if curr_split.is_better_than(best_split):
            best_split = curr_split
    return best_split


def information_gain(tree_node, attrib_index):
    """Gets the attribute's best split according to the binary Information Gain.

    Calculates and returns the Information Gain of the child nodes, without
    taking into account the original information.
    """
    all_values = set(range(tree_node.contingency_tables[attrib_index].shape[0]))
    num_samples = tree_node.dataset.num_samples
    contingency_table = tree_node.contingency_tables[
        attrib_index].contingency_table
    num_samples_per_value = tree_node.contingency_tables[
        attrib_index].num_samples_per_value
    best_split = split.Split()
    for left_values in split.powerset_using_symmetry(all_values):
        right_values = all_values - left_values
        num_left_samples, num_right_samples = split.get_num_samples_per_side(
            num_samples, num_samples_per_value, left_values, right_values)
        split_information_gain = calculate_information_gain(
            num_samples, contingency_table, left_values, right_values,
            num_left_samples, num_right_samples)
        curr_split = split.Split(left_values=left_values,
                                 right_values=right_values,
                                 impurity=split_information_gain)
        if curr_split.is_better_than(best_split):
            best_split = curr_split
    return best_split


def gain_ratio(tree_node, attrib_index):
    """Gets the attribute's best split according to the binary Gain Ratio.

    Calculates and returns the Gain Ratio of the child nodes, without taking
    into account the original information.
    """
    all_values = set(range(tree_node.contingency_tables[attrib_index].shape[0]))
    num_samples = tree_node.dataset.num_samples
    contingency_table = tree_node.contingency_tables[
        attrib_index].contingency_table
    num_samples_per_value = tree_node.contingency_tables[
        attrib_index].num_samples_per_value
    best_split = split.Split()
    for left_values in split.powerset_using_symmetry(all_values):
        right_values = all_values - left_values
        split_gain_ratio = calculate_gain_ratio(num_samples,
                                                contingency_table,
                                                num_samples_per_value,
                                                left_values,
                                                right_values)
        curr_split = split.Split(left_values=left_values,
                                 right_values=right_values,
                                 impurity=split_gain_ratio)
        if curr_split.is_better_than(best_split):
            best_split = curr_split
    return best_split


def calculate_gain_ratio(num_samples, contingency_table, num_samples_per_value,
                         left_values, right_values):
    """Calculates the Gain Ratio of the given binary split."""
    num_left_samples, num_right_samples = split.get_num_samples_per_side(
        num_samples, num_samples_per_value, left_values, right_values)
    split_information_gain = calculate_information_gain(
        num_samples, contingency_table, left_values, right_values,
        num_left_samples, num_right_samples)
    potential_partition_information = calculate_potential_information(
        num_samples, num_left_samples, num_right_samples)
    return split_information_gain / potential_partition_information


def calculate_potential_information(num_samples, num_left_samples,
                                    num_right_samples):
    """Calculates the Potential Information of the given binary split."""
    left_ratio = num_left_samples / num_samples
    right_ratio = num_right_samples / num_samples
    partition_potential_information = (
        - left_ratio * math.log2(left_ratio)
        - right_ratio * math.log2(right_ratio))
    return partition_potential_information

def calculate_information_gain(num_samples, contingency_table, left_values,
                               right_values, num_left_samples,
                               num_right_samples):
    """Calculates the Information Gain of the given binary split."""
    num_samples_per_class_left = split.get_num_samples_per_class_in_values(
        contingency_table, left_values)
    left_split_information = calculate_information(
        num_left_samples, num_samples_per_class_left)
    num_samples_per_class_right = split.get_num_samples_per_class_in_values(
        contingency_table, right_values)
    right_split_information = calculate_information(
        num_right_samples, num_samples_per_class_right)
    split_information_gain = (
        (num_left_samples / num_samples) * left_split_information
        + (num_right_samples / num_samples) * right_split_information)
    return split_information_gain


def calculate_information(num_split_samples, num_samples_per_class_in_split):
    """Calculates the Information of the node given by the values."""
    information = 0.0
    for curr_class_num_samples in num_samples_per_class_in_split:
        if curr_class_num_samples != 0:
            curr_frequency = curr_class_num_samples / num_split_samples
            information -= curr_frequency * math.log2(curr_frequency)
    return information


def sliq(tree_node, attrib_index):
    """Gets the attribute's best split according to the SLIQ criterion."""
    all_values = set(range(tree_node.contingency_tables[attrib_index].shape[0]))
    num_samples = tree_node.dataset.num_samples
    contingency_table = tree_node.contingency_tables[
        attrib_index].contingency_table
    num_samples_per_value = tree_node.contingency_tables[
        attrib_index].num_samples_per_value
    best_split = split.Split(left_values=set(all_values),
                             right_values=set())
    while best_split.left_values:
        iteration_best_split = split.Split()
        for value in best_split.left_values:
            curr_left_values = best_split.left_values - set([value])
            curr_right_values = best_split.right_values + set([value])
            curr_split_gini = calculate_split_gini_index(num_samples,
                                                         contingency_table,
                                                         num_samples_per_value,
                                                         curr_left_values,
                                                         curr_right_values)
            curr_split = split.Split(left_values=curr_left_values,
                                     right_values=curr_right_values,
                                     impurity=curr_split_gini)
            if curr_split.is_better_than(iteration_best_split):
                iteration_best_split = curr_split
        if iteration_best_split.is_better_than(best_split):
            best_split = iteration_best_split
        else:
            break
    return best_split


def sliq_ext(tree_node, attrib_index):
    """Gets the attribute's best split according to the SLIQ-ext criterion."""
    all_values = set(range(tree_node.contingency_tables[attrib_index].shape[0]))
    num_samples = tree_node.dataset.num_samples
    contingency_table = tree_node.contingency_tables[
        attrib_index].contingency_table
    num_samples_per_value = tree_node.contingency_tables[
        attrib_index].num_samples_per_value
    best_split = split.Split()
    iteration_start_split = split.Split(left_values=set(all_values),
                                        right_values=set())
    while iteration_start_split.left_values:
        iteration_best_split = split.Split()
        for value in iteration_start_split.left_values:
            curr_left_values = iteration_start_split.left_values - set([value])
            curr_right_values = (
                iteration_start_split.right_values + set([value]))
            curr_split_gini = calculate_split_gini_index(num_samples,
                                                         contingency_table,
                                                         num_samples_per_value,
                                                         curr_left_values,
                                                         curr_right_values)
            curr_split = split.Split(left_values=curr_left_values,
                                     right_values=curr_right_values,
                                     impurity=curr_split_gini)
            if curr_split.is_better_than(iteration_best_split):
                iteration_best_split = curr_split
        if iteration_best_split.is_better_than(best_split):
            best_split = iteration_best_split
        iteration_start_split = iteration_best_split
    return best_split


def flip_flop(partition_init_fn, tree_node, attrib_index, node_impurity_fn):
    """Gets the attribute's best split according to the Flip-Flop criterion."""
    num_samples = tree_node.dataset.num_samples
    contingency_table = tree_node.contingency_tables[
        attrib_index].contingency_table
    transposed_contingency_table = contingency_table.T
    num_values, num_classes = contingency_table.shape
    num_samples_per_value = tree_node.contingency_tables[
        attrib_index].num_samples_per_value
    num_samples_per_class = split.get_num_samples_per_class(contingency_table)
    left_values, right_values = partition_init_fn(
        tree_node, attrib_index, node_impurity_fn)
    curr_split_gini_index = calculate_split_gini_index(
        num_samples, contingency_table, num_samples_per_value, left_values,
        right_values)
    best_split = split.Split(left_values=left_values,
                             right_values=right_values,
                             impurity=curr_split_gini_index)
    for _ in range(MAX_ITERATIONS):
        # Use values split as 2 supervalues and create binary split of classes.
        smaller_values_set = split.get_smaller_set(best_split.left_values,
                                                   best_split.right_values)
        supervalues_contingency_table = get_contingency_table_for_superclasses(
            num_classes, transposed_contingency_table, num_samples_per_class,
            smaller_values_set)
        curr_classes_split = get_best_split(
            num_samples, supervalues_contingency_table, num_samples_per_class,
            node_impurity_fn)
        # Use classes split as 2 superclasses and create binary split of values.
        smaller_classes_set = split.get_smaller_set(
            curr_classes_split.left_values, curr_classes_split.right_values)
        superclasses_contingency_table = get_contingency_table_for_superclasses(
            num_values, contingency_table, num_samples_per_value,
            smaller_classes_set)
        curr_split = get_best_split(
            num_samples, superclasses_contingency_table, num_samples_per_value,
            node_impurity_fn)
        if curr_split == best_split:
            break
        else:
            best_split = curr_split
    return best_split


def create_random_partition(num_values):
    """Creates a random partition of the integer values in [0, num_values)."""
    left_values = set()
    right_values = set()
    for value in range(num_values):
        if random.choice((0, 1)):
            left_values.add(value)
        else:
            right_values.add(value)
    return left_values, right_values


def init_with_max(tree_node, attrib_index, node_impurity_fn):
    """Generates a split grouping classes in superclasses, largest  one alone.
    """
    def split_max(num_samples_per_index):
        """Returns the index with the largest count."""
        index_of_max, _ = max(enumerate(num_samples_per_index),
                              key=operator.itemgetter(1))
        return index_of_max

    num_samples = tree_node.dataset.num_samples
    contingency_table = tree_node.contingency_tables[
        attrib_index].contingency_table
    num_samples_per_class = split.get_num_samples_per_class(
        contingency_table)
    left_class = split_max(num_samples_per_class)
    num_values = tree_node.contingency_tables[attrib_index].shape[0]
    num_samples_per_value = tree_node.contingency_tables[
        attrib_index].num_samples_per_value
    superclasses_contingency_table = get_contingency_table_for_superclasses(
        num_values, contingency_table, num_samples_per_value, set([left_class]))
    curr_split = get_best_split(
        num_samples, superclasses_contingency_table, num_samples_per_value,
        node_impurity_fn)
    return curr_split.left_values, curr_split.right_values


def init_with_list_scheduling(tree_node, attrib_index, node_impurity_fn):
    """Generates a split grouping classes in superclasses using list scheduling.
    """
    def list_scheduling(num_samples_per_index):
        """Groups indices in 2 groups, balanced using list scheduling."""
        sorted_indices_and_count = get_indices_count_sorted(
            num_samples_per_index)
        left_indices = set()
        left_count = 0
        right_indices = set()
        right_count = 0
        for index, index_num_samples in sorted_indices_and_count:
            if left_count > right_count:
                right_indices.add(index)
                right_count += index_num_samples
            else:
                left_indices.add(index)
                left_count += index_num_samples
        return left_indices, right_indices

    num_samples = tree_node.dataset.num_samples
    contingency_table = tree_node.contingency_tables[
        attrib_index].contingency_table
    num_values = contingency_table.shape[0]
    num_samples_per_class = split.get_num_samples_per_class(
        contingency_table)
    left_classes, _ = list_scheduling(num_samples_per_class)
    num_samples_per_value = tree_node.contingency_tables[
        attrib_index].num_samples_per_value
    num_samples_per_value_from_left_classes = (
        split.get_num_samples_per_value_in_classes(contingency_table,
                                                   left_classes))
    num_samples_per_value_from_right_classes = [
        (num_samples_per_value[value] -
         num_samples_per_value_from_left_classes[value])
        for value in range(num_values)]
    superclasses_contingency_table = np.array(
        [num_samples_per_value_from_left_classes,
         num_samples_per_value_from_right_classes], dtype=int)
    curr_split = get_best_split(num_samples,
                                superclasses_contingency_table,
                                num_samples_per_value,
                                node_impurity_fn)
    return curr_split.left_values, curr_split.right_values
