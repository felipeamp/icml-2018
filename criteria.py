#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Module containing all criteria available for tests."""

import functools
import math
import operator
import random

import numpy as np

# import local_search
import split

MAX_ITERATIONS = 100
NUM_RANDOM_PARTITIONS_TO_TEST = 1


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
                    tree_node=tree_node, attrib_index=attrib_index)
                best_splits.append(attrib_best_split)
            else:
                best_splits.append([])
        return best_splits


def save_superclasses_largest_frequence(num_samples, superclass_contingency_table, best_split):
    """Saves the superclasses_largest_frequence field in best_split."""
    num_samples_per_class = split.get_num_samples_per_class(superclass_contingency_table)
    best_split.set_superclasses_largest_frequence(max(num_samples_per_class) / num_samples)


def get_indices_count_sorted(num_samples_per_index, reverse=False):
    """Returns list of indices and their count, ordered by count, in num_samples_per_value."""
    num_samples_per_index_enumerated = list(
        enumerate(num_samples_per_index))
    num_samples_per_index_enumerated.sort(key=lambda x: x[1], reverse=reverse)
    return num_samples_per_index_enumerated


def get_indices_frequency_sorted(superclass_contingency_table, num_samples_per_value):
    """Returns indices and their frequency, ordered by frequency, in superclass_contingency_table.
    """
    num_values = num_samples_per_value.shape[0]
    # DEBUG
    # if 0 in num_samples_per_value:
    #     print("num_samples_per_value has non-existent value")
    frequency_per_value_enumerated = list(enumerate(
        superclass_contingency_table[value, 0] / num_samples_per_value[value]
        for value in range(num_values)))
    frequency_per_value_enumerated.sort(key=lambda x: x[1])
    return frequency_per_value_enumerated


def get_indices_sorted_per_frequency(superclass_contingency_table, num_samples_per_value):
    """Returns list of values indices ordered by their frequency in class 0."""
    indices_freq_sorted = get_indices_frequency_sorted(superclass_contingency_table,
                                                       num_samples_per_value)
    return [index for (index, _) in indices_freq_sorted]


def calculate_split_gini_index(num_samples, contingency_table,
                               num_samples_per_value, left_values,
                               right_values):
    """Calculates the weighted Gini index of a split."""
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
    # DEBUG:
    print("num_left_samples:", num_left_samples)
    print("num_right_samples:", num_right_samples)
    print("num_samples_per_class_left:", num_samples_per_class_left)
    print("num_samples_per_class_right:", num_samples_per_class_right)
    print("left_gini:", left_gini)
    print("right_gini:", right_gini)
    print("child_gini:", ((num_left_samples / num_samples) * left_gini +
                          (num_right_samples / num_samples) * right_gini))
    return ((num_left_samples / num_samples) * left_gini +
            (num_right_samples / num_samples) * right_gini)


def calculate_node_gini_index(num_split_samples,
                              num_samples_per_class_in_split):
    """Calculates the Gini index of a node."""
    if not num_split_samples:
        return 1.0
    gini_index = 1.0
    for curr_class_num_samples in num_samples_per_class_in_split:
        gini_index -= (curr_class_num_samples / num_split_samples)**2
    return gini_index


def get_best_split(num_samples, contingency_table, num_samples_per_value,
                   node_impurity_fn):
    """Gets the best split using the two-class trick. Assumes contingency_table has only 2 columns.
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
    num_values, num_classes = contingency_table.shape
    values_sorted_per_freq = get_indices_sorted_per_frequency(contingency_table,
                                                              num_samples_per_value)
    # We start with the (invalid) split where every value is on the right side.
    curr_split = split.Split(left_values=set(),
                             right_values=set(range(num_values)))
    best_split = curr_split
    # We use the four variables below in the dynamic programming algorithm to calculate the
    # impurity of a split.
    num_left_samples = 0
    num_right_samples = num_samples
    num_samples_per_class_left = [0] * num_classes
    num_samples_per_class_right = split.get_num_samples_per_class(contingency_table)
    for last_left_value in values_sorted_per_freq[:-1]:
        left_values = curr_split.left_values | set([last_left_value])
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
    all_values = set(range(tree_node.contingency_tables[attrib_index].contingency_table.shape[0]))
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
            right_values = all_values - set(left_values)
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
         num_samples_per_value_from_right_classes], dtype=int).T
    return superclasses_contingency_table


def twoing_superclass_partition(tree_node, attrib_index, node_impurity_fn):
    """Gets the attribute's best split according to the Twoing criterion."""
    num_values, num_classes = tree_node.contingency_tables[attrib_index].contingency_table.shape
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
                                    node_impurity_fn)
        if curr_split.is_better_than(best_split):
            best_split = curr_split
            save_superclasses_largest_frequence(
                num_samples, superclasses_contingency_table, best_split)
    return best_split


def twoing_k_class_partition(tree_node, attrib_index, node_impurity_fn):
    """Gets the attribute's best split according to the Twoing criterion."""
    num_values, num_classes = tree_node.contingency_tables[attrib_index].contingency_table.shape
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
        # DEBUG
        # print("left_classes:", left_classes)
        curr_split = get_best_split_2(num_samples,
                                      superclasses_contingency_table,
                                      contingency_table,
                                      num_samples_per_value,
                                      node_impurity_fn)
        if curr_split.is_better_than(best_split):
            best_split = curr_split
            save_superclasses_largest_frequence(
                num_samples, superclasses_contingency_table, best_split)
    # DEBUG
    # print("twoing_k_class_partition")
    # print("contingency_table")
    # print(contingency_table)
    # print("num_samples_per_value:", num_samples_per_value)
    # print("best_split.left_values:", best_split.left_values)
    # print("best_split.right_values:", best_split.right_values)
    # print("best_split.impurity:", best_split.impurity)
    return best_split


def information_gain(tree_node, attrib_index):
    """Gets the attribute's best split according to the binary Information Gain.

    Calculates and returns the Information Gain of the child nodes, without
    taking into account the original information.
    """
    all_values = set(range(tree_node.contingency_tables[attrib_index].contingency_table.shape[0]))
    num_samples = tree_node.dataset.num_samples
    contingency_table = tree_node.contingency_tables[
        attrib_index].contingency_table
    num_samples_per_value = tree_node.contingency_tables[
        attrib_index].num_samples_per_value
    best_split = split.Split()
    for left_values in split.powerset_using_symmetry(all_values):
        right_values = all_values - set(left_values)
        split_information_gain = calculate_information_gain(
            num_samples, contingency_table, num_samples_per_value, left_values, right_values)
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
    all_values = set(range(tree_node.contingency_tables[attrib_index].contingency_table.shape[0]))
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
    split_information_gain = calculate_information_gain(
        num_samples, contingency_table, num_samples_per_value, left_values, right_values)
    potential_partition_information = calculate_potential_information(
        num_samples, num_samples_per_value, left_values, right_values)
    return split_information_gain / potential_partition_information


def calculate_potential_information(num_samples, num_samples_per_value, left_values, right_values):
    """Calculates the Potential Information of the given binary split."""
    num_left_samples, num_right_samples = split.get_num_samples_per_side(
        num_samples, num_samples_per_value, left_values, right_values)
    left_ratio = num_left_samples / num_samples
    right_ratio = num_right_samples / num_samples
    partition_potential_information = (
        - left_ratio * math.log2(left_ratio)
        - right_ratio * math.log2(right_ratio))
    return partition_potential_information

def calculate_information_gain(num_samples, contingency_table, num_samples_per_value, left_values,
                               right_values):
    """Calculates the Information Gain of the given binary split."""
    num_left_samples, num_right_samples = split.get_num_samples_per_side(
        num_samples, num_samples_per_value, left_values, right_values)
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
    all_values = set(range(tree_node.contingency_tables[attrib_index].contingency_table.shape[0]))
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
            curr_right_values = best_split.right_values | set([value])
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


def sliq_ext(tree_node, attrib_index, split_impurity_fn):
    """Gets the attribute's best split according to the SLIQ-ext criterion."""
    all_values = set(range(tree_node.contingency_tables[attrib_index].contingency_table.shape[0]))
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
                iteration_start_split.right_values | set([value]))
            curr_split_impurity = split_impurity_fn(num_samples,
                                                    contingency_table,
                                                    num_samples_per_value,
                                                    curr_left_values,
                                                    curr_right_values)
            curr_split = split.Split(left_values=curr_left_values,
                                     right_values=curr_right_values,
                                     impurity=curr_split_impurity)
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
    transposed_contingency_table = np.copy(contingency_table).T
    num_values, num_classes = contingency_table.shape
    num_samples_per_value = tree_node.contingency_tables[
        attrib_index].num_samples_per_value
    num_samples_per_class = split.get_num_samples_per_class(contingency_table)
    best_split = partition_init_fn(tree_node, attrib_index, node_impurity_fn)
    for iteration_number in range(1, MAX_ITERATIONS + 1):
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
            num_values, contingency_table, num_samples_per_value, smaller_classes_set)
        curr_split = get_best_split(
            num_samples, superclasses_contingency_table, num_samples_per_value, node_impurity_fn)
        converged = curr_split == best_split
        best_split = curr_split
        best_split.set_iteration_number(iteration_number)
        if converged:
            break
    return best_split


def flip_flop_2(partition_init_fn, tree_node, attrib_index, node_impurity_fn):
    """Gets the attribute's best split according to the Flip-Flop criterion."""
    num_samples = tree_node.dataset.num_samples
    contingency_table = tree_node.contingency_tables[
        attrib_index].contingency_table
    transposed_contingency_table = np.copy(contingency_table).T
    num_values, num_classes = contingency_table.shape
    num_samples_per_value = tree_node.contingency_tables[
        attrib_index].num_samples_per_value
    num_samples_per_class = split.get_num_samples_per_class(contingency_table)
    best_split = partition_init_fn(tree_node, attrib_index, node_impurity_fn)
    for iteration_number in range(1, MAX_ITERATIONS + 1):
        # Use values split as 2 supervalues and create binary split of classes.
        smaller_values_set = split.get_smaller_set(best_split.left_values,
                                                   best_split.right_values)
        supervalues_contingency_table = get_contingency_table_for_superclasses(
            num_classes, transposed_contingency_table, num_samples_per_class,
            smaller_values_set)
        curr_classes_split = get_best_split_2(
            num_samples, supervalues_contingency_table, transposed_contingency_table,
            num_samples_per_class, node_impurity_fn)
        # Use classes split as 2 superclasses and create binary split of values.
        smaller_classes_set = split.get_smaller_set(
            curr_classes_split.left_values, curr_classes_split.right_values)
        superclasses_contingency_table = get_contingency_table_for_superclasses(
            num_values, contingency_table, num_samples_per_value, smaller_classes_set)
        curr_split = get_best_split_2(
            num_samples, superclasses_contingency_table, contingency_table, num_samples_per_value,
            node_impurity_fn)
        converged = curr_split == best_split
        best_split = curr_split
        best_split.set_iteration_number(iteration_number)
        if converged:
            break
    return best_split


def create_nonempty_random_partition(num_values):
    """Creates a random partition with at least one value in each side."""
    assert num_values >= 2
    while True:
        left_values = set()
        right_values = set()
        for value in range(num_values):
            if random.getrandbits(1):
                left_values.add(value)
            else:
                right_values.add(value)
        if left_values and right_values:
            # avoids empty-full partitions
            break
    if (len(left_values) > len(right_values) or
            (len(left_values) == len(right_values) and 0 in right_values)):
        left_values, right_values = right_values, left_values
    return left_values, right_values


def create_values_random_partition(tree_node, attrib_index, _,
                                   num_partitions_to_test=NUM_RANDOM_PARTITIONS_TO_TEST):
    """Get best random partition of [0, num_values) among num_partitions_to_test of them."""
    num_values = tree_node.contingency_tables[attrib_index].contingency_table.shape[0]
    assert num_partitions_to_test <= 2 ** (num_values - 1) - 1
    left_partitions_seen = set()
    best_split = split.Split()
    for _ in range(num_partitions_to_test):
        while True:
            left_values, right_values = create_nonempty_random_partition(num_values)
            left_values_as_str = ','.join(map(str, sorted(left_values)))
            if left_values_as_str not in left_partitions_seen:
                left_partitions_seen.add(left_values_as_str)
                break
        curr_split = split.Split(left_values=left_values,
                                 right_values=right_values)
        if curr_split.is_better_than(best_split):
            best_split = curr_split
    return best_split


def create_classes_random_partition(num_classes):
    """Creates a random partition of the integer classes in [0, num_classes)."""
    return create_nonempty_random_partition(num_classes)


def init_with_largest_alone_superclass_partition(tree_node, attrib_index, node_impurity_fn):
    """Generates a split grouping classes in superclasses, largest one alone.
    """
    def get_index_of_largest(num_samples_per_index):
        """Returns the index with the largest count."""
        index_of_max, _ = max(enumerate(num_samples_per_index),
                              key=operator.itemgetter(1))
        return index_of_max

    num_samples = tree_node.dataset.num_samples
    contingency_table = tree_node.contingency_tables[
        attrib_index].contingency_table
    num_samples_per_class = split.get_num_samples_per_class(contingency_table)
    left_class = get_index_of_largest(num_samples_per_class)
    num_values = tree_node.contingency_tables[attrib_index].contingency_table.shape[0]
    num_samples_per_value = tree_node.contingency_tables[
        attrib_index].num_samples_per_value
    superclasses_contingency_table = get_contingency_table_for_superclasses(
        num_values, contingency_table, num_samples_per_value, set([left_class]))
    curr_split = get_best_split(
        num_samples, superclasses_contingency_table, num_samples_per_value,
        node_impurity_fn)
    save_superclasses_largest_frequence(
        num_samples, superclasses_contingency_table, curr_split)
    return curr_split


def init_with_largest_alone_k_class_partition(tree_node, attrib_index, node_impurity_fn):
    """Generates a split grouping classes in superclasses, largest one alone.
    """
    def get_index_of_largest(num_samples_per_index):
        """Returns the index with the largest count."""
        index_of_max, _ = max(enumerate(num_samples_per_index), key=operator.itemgetter(1))
        return index_of_max

    num_samples = tree_node.dataset.num_samples
    contingency_table = tree_node.contingency_tables[
        attrib_index].contingency_table
    num_samples_per_class = split.get_num_samples_per_class(contingency_table)
    left_class = get_index_of_largest(num_samples_per_class)
    num_values = tree_node.contingency_tables[attrib_index].contingency_table.shape[0]
    num_samples_per_value = tree_node.contingency_tables[
        attrib_index].num_samples_per_value
    superclasses_contingency_table = get_contingency_table_for_superclasses(
        num_values, contingency_table, num_samples_per_value, set([left_class]))
    # DEBUG
    # print("init_with_largest_alone_k_class_partition")
    # print("contingency_table")
    # print(contingency_table)
    # print("num_samples_per_value:", num_samples_per_value)
    # print("left_class:", left_class)
    # print("superclasses_contingency_table")
    curr_split = get_best_split_2(
        num_samples, superclasses_contingency_table, contingency_table, num_samples_per_value,
        node_impurity_fn)
    save_superclasses_largest_frequence(
        num_samples, superclasses_contingency_table, curr_split)
    # DEBUG
    # print("curr_split.left_values:", curr_split.left_values)
    # print("curr_split.right_values:", curr_split.right_values)
    # print("curr_split.impurity:", curr_split.impurity)
    return curr_split


def init_with_list_scheduling_superclass_partition(tree_node, attrib_index, node_impurity_fn):
    """Generates a split grouping classes in superclasses using list scheduling.
    """
    def list_scheduling(num_samples_per_index):
        """Groups indices in 2 groups, balanced using list scheduling."""
        rev_sorted_indices_and_count = get_indices_count_sorted(num_samples_per_index, reverse=True)
        left_indices = set([rev_sorted_indices_and_count[0][0]])
        left_count = rev_sorted_indices_and_count[0][1]
        right_indices = set()
        right_count = 0
        for index, index_num_samples in rev_sorted_indices_and_count[1:]:
            if left_count >= right_count:
                right_indices.add(index)
                right_count += index_num_samples
            else:
                left_indices.add(index)
                left_count += index_num_samples
        if (len(left_indices) > len(right_indices) or
                (len(left_indices) == len(right_indices) and 0 in right_indices)):
            left_indices, right_indices = right_indices, left_indices
        return left_indices, right_indices

    num_samples = tree_node.dataset.num_samples
    contingency_table = tree_node.contingency_tables[
        attrib_index].contingency_table
    num_samples_per_class = split.get_num_samples_per_class(contingency_table)
    left_classes, _ = list_scheduling(num_samples_per_class)
    num_values = contingency_table.shape[0]
    num_samples_per_value = tree_node.contingency_tables[attrib_index].num_samples_per_value
    superclasses_contingency_table = get_contingency_table_for_superclasses(
        num_values, contingency_table, num_samples_per_value, left_classes)
    curr_split = get_best_split(num_samples,
                                superclasses_contingency_table,
                                num_samples_per_value,
                                node_impurity_fn)
    save_superclasses_largest_frequence(
        num_samples, superclasses_contingency_table, curr_split)
    return curr_split



def init_with_list_scheduling_k_class_partition(tree_node, attrib_index, node_impurity_fn):
    """Generates a split grouping classes in superclasses using list scheduling.
    """
    def list_scheduling(num_samples_per_index):
        """Groups indices in 2 groups, balanced using list scheduling."""
        rev_sorted_indices_and_count = get_indices_count_sorted(num_samples_per_index, reverse=True)
        # DEBUG
        # print("rev_sorted_indices_and_count:", rev_sorted_indices_and_count)
        left_indices = set([rev_sorted_indices_and_count[0][0]])
        left_count = rev_sorted_indices_and_count[0][1]
        right_indices = set()
        right_count = 0
        for index, index_num_samples in rev_sorted_indices_and_count[1:]:
            if left_count >= right_count:
                right_indices.add(index)
                right_count += index_num_samples
            else:
                left_indices.add(index)
                left_count += index_num_samples
        if (len(left_indices) > len(right_indices) or
                (len(left_indices) == len(right_indices) and 0 in right_indices)):
            left_indices, right_indices = right_indices, left_indices
        return left_indices, right_indices

    num_samples = tree_node.dataset.num_samples
    contingency_table = tree_node.contingency_tables[
        attrib_index].contingency_table
    num_samples_per_class = split.get_num_samples_per_class(contingency_table)
    left_classes, _ = list_scheduling(num_samples_per_class)
    num_values = contingency_table.shape[0]
    num_samples_per_value = tree_node.contingency_tables[attrib_index].num_samples_per_value
    superclasses_contingency_table = get_contingency_table_for_superclasses(
        num_values, contingency_table, num_samples_per_value, left_classes)
    # DEBUG
    # print("init_with_list_scheduling_k_class_partition")
    # print("contingency_table")
    # print(contingency_table)
    # print("num_samples_per_value:", num_samples_per_value)
    # print("left_classes:", left_classes)
    # print("superclasses_contingency_table")
    curr_split = get_best_split_2(num_samples,
                                  superclasses_contingency_table,
                                  contingency_table,
                                  num_samples_per_value,
                                  node_impurity_fn)
    save_superclasses_largest_frequence(
        num_samples, superclasses_contingency_table, curr_split)
    # DEBUG
    # print("curr_split.left_values:", curr_split.left_values)
    # print("curr_split.right_values:", curr_split.right_values)
    # print("curr_split.impurity:", curr_split.impurity)
    return curr_split


def get_best_split_2(num_samples, superclass_contingency_table, contingency_table,
                     num_samples_per_value, node_impurity_fn):
    """Gets the best split using the two-class trick. Assumes contingency_table has only 2 columns.
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

    assert superclass_contingency_table.shape[1] == 2
    num_values, num_classes = contingency_table.shape
    values_sorted_per_freq = get_indices_sorted_per_frequency(superclass_contingency_table,
                                                              num_samples_per_value)
    # DEBUG:
    # print("values_sorted_per_freq:", values_sorted_per_freq)
    # We start with the (invalid) split where every value is on the right side.
    curr_split = split.Split(left_values=set(),
                             right_values=set(range(num_values)))
    best_split = curr_split
    # We use the four variables below in the dynamic programming algorithm to calculate the
    # impurity of a split.
    num_left_samples = 0
    num_right_samples = num_samples
    num_samples_per_class_left = [0] * num_classes
    num_samples_per_class_right = split.get_num_samples_per_class(contingency_table)
    for last_left_value in values_sorted_per_freq[:-1]:
        left_values = curr_split.left_values | set([last_left_value])
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
        # DEBUG:
        # print("\tleft_values:", left_values)
        # print("\tright_values:", right_values)
        # print("\tsplit_impurity:", split_impurity)
        if curr_split.is_better_than(best_split):
            best_split = curr_split
    return best_split


def random_class_partition_superclass_partition(tree_node, attrib_index, node_impurity_fn):
    """Generates random superclass partition and chooses best split based on it."""
    num_samples = tree_node.dataset.num_samples
    contingency_table = tree_node.contingency_tables[
        attrib_index].contingency_table
    num_values, num_classes = contingency_table.shape
    num_samples_per_value = tree_node.contingency_tables[
        attrib_index].num_samples_per_value
    left_classes, _ = create_classes_random_partition(num_classes)
    superclasses_contingency_table = get_contingency_table_for_superclasses(
        num_values, contingency_table, num_samples_per_value, left_classes)
    best_split = get_best_split(
        num_samples, superclasses_contingency_table, num_samples_per_value, node_impurity_fn)
    save_superclasses_largest_frequence(
        num_samples, superclasses_contingency_table, best_split)
    return best_split


def random_class_partition_k_class_partition(tree_node, attrib_index, node_impurity_fn):
    """Generates random superclass partition. Gets valid splits based on it."""
    num_samples = tree_node.dataset.num_samples
    contingency_table = tree_node.contingency_tables[
        attrib_index].contingency_table
    num_values, num_classes = contingency_table.shape
    num_samples_per_value = tree_node.contingency_tables[
        attrib_index].num_samples_per_value
    left_classes, _ = create_classes_random_partition(num_classes)
    superclasses_contingency_table = get_contingency_table_for_superclasses(
        num_values, contingency_table, num_samples_per_value, left_classes)
    # DEBUG
    # print("random_class_partition_k_class_partition")
    # print("contingency_table")
    # print(contingency_table)
    # print("num_samples_per_value:", num_samples_per_value)
    # print("left_classes:", left_classes)
    # print("superclasses_contingency_table")
    # print(superclasses_contingency_table)
    best_split = get_best_split_2(num_samples, superclasses_contingency_table, contingency_table,
                                  num_samples_per_value, node_impurity_fn)
    save_superclasses_largest_frequence(
        num_samples, superclasses_contingency_table, best_split)
    # DEBUG
    # print("num_samples_per_value:", num_samples_per_value)
    # print("best_split.left_values:", best_split.left_values)
    # print("best_split.right_values:", best_split.right_values)
    # print("best_split.impurity:", best_split.impurity)
    return best_split


def group_values(contingency_table, num_samples_per_value):
    """Groups values that have the same class probability vector."""
    prob_matrix_transposed = np.divide(contingency_table.T, num_samples_per_value)
    prob_matrix = prob_matrix_transposed.T
    row_order = np.lexsort(prob_matrix_transposed[::-1])
    compared_index = row_order[0]
    new_index_to_old = [[compared_index]]
    for index in row_order[1:]:
        if np.allclose(prob_matrix[compared_index], prob_matrix[index]):
            new_index_to_old[-1].append(index)
        else:
            compared_index = index
            new_index_to_old.append([compared_index])
    new_num_values = len(new_index_to_old)
    num_classes = contingency_table.shape[1]
    new_contingency_table = np.zeros((new_num_values, num_classes), dtype=int)
    new_num_samples_per_value = np.zeros((new_num_values), dtype=int)
    for new_index, old_indices in enumerate(new_index_to_old):
        new_contingency_table[new_index] = np.sum(contingency_table[old_indices, :], axis=0)
        new_num_samples_per_value[new_index] = np.sum(num_samples_per_value[old_indices])
    return new_contingency_table, new_num_samples_per_value, new_index_to_old


def change_split_to_use_old_values(best_split_with_new_values, new_index_to_old):
    """Change split values to use indices of original contingency table."""
    left_old_values = set()
    for new_index in best_split_with_new_values.left_values:
        left_old_values |= set(new_index_to_old[new_index])
    right_old_values = set()
    for new_index in best_split_with_new_values.right_values:
        right_old_values |= set(new_index_to_old[new_index])
    best_split_with_new_values.left_values = left_old_values
    best_split_with_new_values.right_values = right_old_values


def get_principal_component(num_samples, contingency_table, num_samples_per_value):
    """Returns the principal component of the weighted covariance matrix."""
    num_samples_per_class = split.get_num_samples_per_class(contingency_table)
    avg_prob_per_class = np.divide(num_samples_per_class, num_samples)
    prob_matrix = contingency_table / num_samples_per_value[:, None]
    diff_prob_matrix = (prob_matrix - avg_prob_per_class).T
    weight_diff_prob = diff_prob_matrix * num_samples_per_value[None, :]
    weighted_squared_diff_prob_matrix = np.dot(weight_diff_prob, diff_prob_matrix.T)
    weighted_covariance_matrix = (1/(num_samples - 1)) * weighted_squared_diff_prob_matrix
    eigenvalues, eigenvectors = np.linalg.eigh(weighted_covariance_matrix)
    index_largest_eigenvalue = np.argmax(np.square(eigenvalues))
    return eigenvectors[:, index_largest_eigenvalue]


def pc(tree_node, attrib_index, split_impurity_fn):
    """Generates partition based on the PC criterion."""
    num_samples = tree_node.dataset.num_samples
    contingency_table = tree_node.contingency_tables[attrib_index].contingency_table
    num_samples_per_value = tree_node.contingency_tables[attrib_index].num_samples_per_value
    (new_contingency_table,
     new_num_samples_per_value,
     new_index_to_old) = group_values(contingency_table, num_samples_per_value)

    principal_component = get_principal_component(
        num_samples, new_contingency_table, new_num_samples_per_value)
    inner_product_results = np.dot(principal_component, new_contingency_table.T)
    new_indices_order = inner_product_results.argsort()

    best_split = split.Split()
    left_values = set()
    right_values = set(new_indices_order)
    for first_right in new_indices_order:
        curr_split_impurity = split_impurity_fn(num_samples, new_contingency_table,
                                                new_num_samples_per_value, left_values,
                                                right_values)
        if curr_split_impurity < best_split.impurity:
            best_split = split.Split(left_values=set(left_values),
                                     right_values=set(right_values),
                                     impurity=curr_split_impurity)
        right_values.remove(first_right)
        left_values.add(first_right)
    change_split_to_use_old_values(best_split, new_index_to_old)
    return best_split


def pc_ext(tree_node, attrib_index, split_impurity_fn):
    """Generates partition based on the PC-ext criterion."""
    num_samples = tree_node.dataset.num_samples
    contingency_table = tree_node.contingency_tables[attrib_index].contingency_table
    num_samples_per_value = tree_node.contingency_tables[attrib_index].num_samples_per_value
    (new_contingency_table,
     new_num_samples_per_value,
     new_index_to_old) = group_values(contingency_table, num_samples_per_value)

    principal_component = get_principal_component(
        num_samples, new_contingency_table, new_num_samples_per_value)
    inner_product_results = np.dot(principal_component, new_contingency_table.T)
    new_indices_order = inner_product_results.argsort()
    # DEBUG
    print("contingency_table:", contingency_table)
    print("values_num_samples:", num_samples_per_value)
    print("new_contingency_table:", new_contingency_table)
    print("new_num_samples_per_value:", new_num_samples_per_value)
    print("new_index_to_old:", new_index_to_old)
    print("principal_component:", principal_component)
    print("inner_product_results:", inner_product_results)
    print("new_indices_order:", new_indices_order)

    best_split = split.Split()
    left_values = set()
    right_values = set(new_indices_order)
    for metaindex, first_right in enumerate(new_indices_order):
        curr_split_impurity = split_impurity_fn(num_samples, new_contingency_table,
                                                new_num_samples_per_value, left_values,
                                                right_values)
        if curr_split_impurity < best_split.impurity:
            best_split = split.Split(left_values=set(left_values),
                                     right_values=set(right_values),
                                     impurity=curr_split_impurity)
        if left_values: # extended splits
            last_left = new_indices_order[metaindex - 1]
            left_values.remove(last_left)
            right_values.add(last_left)
            right_values.remove(first_right)
            left_values.add(first_right)
            curr_ext_split_impurity = split_impurity_fn(num_samples, new_contingency_table,
                                                        new_num_samples_per_value, left_values,
                                                        right_values)
            if curr_ext_split_impurity < best_split.impurity:
                best_split = split.Split(left_values=set(left_values),
                                         right_values=set(right_values),
                                         impurity=curr_ext_split_impurity)
            right_values.remove(last_left)
            left_values.add(last_left)
            left_values.remove(first_right)
            right_values.add(first_right)
        right_values.remove(first_right)
        left_values.add(first_right)
    change_split_to_use_old_values(best_split, new_index_to_old)
    return best_split


RANDOM_FLIPFLOP_GINI = Criterion(
    "Random-FlipFlop-Gini",
    functools.partial(flip_flop,
                      partition_init_fn=create_values_random_partition,
                      node_impurity_fn=calculate_node_gini_index))


LARGEST_ALONE_SUPERCLASS_FLIPFLOP_GINI = Criterion(
    "LargestAloneSuperclass-FlipFlop-Gini",
    functools.partial(flip_flop,
                      partition_init_fn=init_with_largest_alone_superclass_partition,
                      node_impurity_fn=calculate_node_gini_index))


LARGEST_ALONE_K_CLASS_FLIPFLOP_GINI = Criterion(
    "LargestAloneKClass-FlipFlop-Gini",
    functools.partial(flip_flop,
                      partition_init_fn=init_with_largest_alone_k_class_partition,
                      node_impurity_fn=calculate_node_gini_index))


LIST_SCHEDULING_SUPERCLASS_FLIPFLOP_GINI = Criterion(
    "ListSchedulingSuperclass-FlipFlop-Gini",
    functools.partial(flip_flop,
                      partition_init_fn=init_with_list_scheduling_superclass_partition,
                      node_impurity_fn=calculate_node_gini_index))


LIST_SCHEDULING_K_CLASS_FLIPFLOP_GINI = Criterion(
    "ListSchedulingKClass-FlipFlop-Gini",
    functools.partial(flip_flop,
                      partition_init_fn=init_with_list_scheduling_k_class_partition,
                      node_impurity_fn=calculate_node_gini_index))

RANDOM_CLASS_PARTITION_SUPERCLASS_GINI = Criterion(
    "RandomClassPartitionSuperclass-Gini",
    functools.partial(random_class_partition_superclass_partition,
                      node_impurity_fn=calculate_node_gini_index))


RANDOM_CLASS_PARTITION_K_CLASS_GINI = Criterion(
    "RandomClassPartitionKClass-Gini",
    functools.partial(random_class_partition_k_class_partition,
                      node_impurity_fn=calculate_node_gini_index))


TWOING_SUPERCLASS_GINI = Criterion(
    "TwoingSuperclass-Gini",
    functools.partial(twoing_superclass_partition,
                      node_impurity_fn=calculate_node_gini_index))


TWOING_K_CLASS_GINI = Criterion(
    "TwoingKClass-Gini",
    functools.partial(twoing_k_class_partition,
                      node_impurity_fn=calculate_node_gini_index))


GINI_GAIN = Criterion("GiniGain", gini_gain)


RANDOM_FLIPFLOP_ENTROPY = Criterion(
    "Random-FlipFlop-Entropy",
    functools.partial(flip_flop,
                      partition_init_fn=create_values_random_partition,
                      node_impurity_fn=calculate_information))


LIST_SCHEDULING_SUPERCLASS_FLIPFLOP_ENTROPY = Criterion(
    "ListSchedulingSuperclass-FlipFlop-Entropy",
    functools.partial(flip_flop,
                      partition_init_fn=init_with_list_scheduling_superclass_partition,
                      node_impurity_fn=calculate_information))


LIST_SCHEDULING_K_CLASS_FLIPFLOP_ENTROPY = Criterion(
    "ListSchedulingKClass-FlipFlop-Entropy",
    functools.partial(flip_flop,
                      partition_init_fn=init_with_list_scheduling_k_class_partition,
                      node_impurity_fn=calculate_information))


LARGEST_ALONE_SUPERCLASS_FLIPFLOP_ENTROPY = Criterion(
    "LargestAloneSuperclass-FlipFlop-Entropy",
    functools.partial(flip_flop,
                      partition_init_fn=init_with_largest_alone_superclass_partition,
                      node_impurity_fn=calculate_information))


LARGEST_ALONE_K_CLASS_FLIPFLOP_ENTROPY = Criterion(
    "LargestAloneKClass-FlipFlop-Entropy",
    functools.partial(flip_flop,
                      partition_init_fn=init_with_largest_alone_k_class_partition,
                      node_impurity_fn=calculate_information))

RANDOM_CLASS_PARTITION_SUPERCLASS_ENTROPY = Criterion(
    "RandomClassPartitionSuperclass-Entropy",
    functools.partial(random_class_partition_superclass_partition,
                      node_impurity_fn=calculate_information))


RANDOM_CLASS_PARTITION_K_CLASS_ENTROPY = Criterion(
    "RandomClassPartitionKClass-Entropy",
    functools.partial(random_class_partition_k_class_partition,
                      node_impurity_fn=calculate_information))


TWOING_SUPERCLASS_ENTROPY = Criterion(
    "TwoingSuperclass-Entropy",
    functools.partial(twoing_superclass_partition,
                      node_impurity_fn=calculate_information))

TWOING_K_CLASS_ENTROPY = Criterion(
    "TwoingKClass-Entropy",
    functools.partial(twoing_k_class_partition,
                      node_impurity_fn=calculate_information))


INFORMATION_GAIN = Criterion("InformationGain", information_gain)


SLIQ_EXT_GINI = Criterion(
    "SliqExt-Gini",
    functools.partial(sliq_ext,
                      split_impurity_fn=calculate_split_gini_index))


SLIQ_EXT_ENTROPY = Criterion(
    "SliqExt-Entropy",
    functools.partial(sliq_ext,
                      split_impurity_fn=calculate_information_gain))


PC_GINI = Criterion(
    "PC-Gini",
    functools.partial(pc, split_impurity_fn=calculate_split_gini_index))


PC_ENTROPY = Criterion(
    "PC-Entropy",
    functools.partial(pc, split_impurity_fn=calculate_information_gain))


PC_EXT_GINI = Criterion(
    "PCExt-Gini",
    functools.partial(pc_ext,
                      split_impurity_fn=calculate_split_gini_index))


PC_EXT_ENTROPY = Criterion(
    "PCExt-Entropy",
    functools.partial(pc_ext,
                      split_impurity_fn=calculate_information_gain))
