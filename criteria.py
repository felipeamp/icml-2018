#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Module containing all criteria available for tests."""

import itertools
import math

# import local_search
import split


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


def calculate_node_gini_index(num_samples, num_samples_per_class):
    """Calculates the Gini index of a node."""
    gini_index = 1.0
    for curr_class_num_samples in num_samples_per_class:
        gini_index -= (curr_class_num_samples / num_samples)**2
    return gini_index


def calculate_split_gini_index(num_samples, contingency_table,
                               num_samples_per_value, left_values,
                               right_values):
    """Calculates the weigthed Gini index of a split."""
    num_left_samples, num_right_samples = split.get_num_samples_per_side(
        num_samples, num_samples_per_value, left_values, right_values)
    num_samples_per_class_left = split.get_num_samples_per_class(
        contingency_table, left_values)
    left_gini = calculate_node_gini_index(num_samples,
                                          num_samples_per_class_left)
    num_samples_per_class_right = split.get_num_samples_per_class(
        contingency_table, right_values)
    right_gini = calculate_node_gini_index(num_samples,
                                           num_samples_per_class_right)
    return ((num_left_samples / num_samples) * left_gini +
            (num_right_samples / num_samples) * right_gini)


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
    best_split = split.Split()
    for left_values in powerset_using_symmetry(all_values):
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
    for left_values in powerset_using_symmetry(all_values):
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
    for left_values in powerset_using_symmetry(all_values):
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
    left_split_information = calculate_information(
        contingency_table, left_values, num_left_samples)
    right_split_information = calculate_information(
        contingency_table, right_values,
        num_right_samples)
    split_information_gain = (
        - (num_left_samples / num_samples) * left_split_information
        - (num_right_samples / num_samples) * right_split_information)
    return split_information_gain


def calculate_information(contingency_table, values, num_split_samples):
    """Calculates the Information of the node given by the values."""
    num_samples_per_class_split = split.get_num_samples_per_class(
        contingency_table, values)
    information = 0.0
    for curr_class_num_samples in num_samples_per_class_split:
        if curr_class_num_samples != 0:
            curr_frequency = curr_class_num_samples / num_split_samples
            information -= curr_frequency * math.log2(curr_frequency)
    return information





def sliq(tree_node, attrib_index):
    """Gets the attribute's best split according to the SLIQ."""
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
    """Gets the attribute's best split according to the SLIQ-ext."""
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



# #################################################################################################
# #################################################################################################
# ###                                                                                           ###
# ###                                       TWOING                                              ###
# ###                                                                                           ###
# #################################################################################################
# #################################################################################################

# class Twoing(Criterion):
#     """Twoing criterion. For reference see "Breiman, L., Friedman, J. J., Olshen, R. A., and
#     Stone, C. J. Classification and Regression Trees. Wadsworth, 1984".
#     """
#     name = 'Twoing'

#     @classmethod
#     def select_best_attribute_and_split(cls, tree_node):
#         """Returns the best attribute and its best split, according to the Twoing criterion.

#         Args:
#           tree_node (TreeNode): tree node where we want to find the best attribute/split.

#         Returns the best split found.
#         """
#         best_splits_per_attrib = []
#         for (attrib_index,
#              (is_valid_nominal_attrib,
#               is_valid_numeric_attrib)) in enumerate(zip(tree_node.valid_nominal_attribute,
#                                                          tree_node.valid_numeric_attribute)):
#             if is_valid_nominal_attrib:
#                 best_total_gini_gain = float('-inf')
#                 best_left_values = set()
#                 best_right_values = set()
#                 values_seen = cls._get_values_seen(
#                     tree_node.contingency_tables[attrib_index].values_num_samples)
#                 for (set_left_classes,
#                      set_right_classes) in cls._generate_twoing(tree_node.class_index_num_samples):
#                     (twoing_contingency_table,
#                      superclass_index_num_samples) = cls._get_twoing_contingency_table(
#                          tree_node.contingency_tables[attrib_index].contingency_table,
#                          tree_node.contingency_tables[attrib_index].values_num_samples,
#                          set_left_classes,
#                          set_right_classes)
#                     original_gini = cls._calculate_gini_index(len(tree_node.valid_samples_indices),
#                                                               superclass_index_num_samples)
#                     (curr_gini_gain,
#                      left_values,
#                      right_values) = cls._two_class_trick(
#                          original_gini,
#                          superclass_index_num_samples,
#                          values_seen,
#                          tree_node.contingency_tables[attrib_index].values_num_samples,
#                          twoing_contingency_table,
#                          len(tree_node.valid_samples_indices))
#                     if curr_gini_gain > best_total_gini_gain:
#                         best_total_gini_gain = curr_gini_gain
#                         best_left_values = left_values
#                         best_right_values = right_values
#                 best_splits_per_attrib.append(
#                     Split(attrib_index=attrib_index,
#                           splits_values=[best_left_values, best_right_values],
#                           criterion_value=best_total_gini_gain))
#             elif is_valid_numeric_attrib:
#                 values_and_classes = cls._get_numeric_values_seen(tree_node.valid_samples_indices,
#                                                                   tree_node.dataset.samples,
#                                                                   tree_node.dataset.sample_class,
#                                                                   attrib_index)
#                 values_and_classes.sort()
#                 (best_twoing,
#                  last_left_value,
#                  first_right_value) = cls._twoing_for_numeric(
#                      values_and_classes,
#                      tree_node.dataset.num_classes)
#                 best_splits_per_attrib.append(
#                     Split(attrib_index=attrib_index,
#                           splits_values=[{last_left_value}, {first_right_value}],
#                           criterion_value=best_twoing))
#         if best_splits_per_attrib:
#             return max(best_splits_per_attrib, key=lambda split: split.criterion_value)
#         return Split()

#     @staticmethod
#     def _generate_twoing(class_index_num_samples):
#         # We only need to look at superclasses of up to (len(class_index_num_samples)/2 + 1)
#         # elements because of symmetry! The subsets we are not choosing are complements of the ones
#         # chosen.
#         non_empty_classes = set([])
#         for class_index, class_num_samples in enumerate(class_index_num_samples):
#             if class_num_samples > 0:
#                 non_empty_classes.add(class_index)
#         number_non_empty_classes = len(non_empty_classes)

#         for left_classes in itertools.chain.from_iterable(
#                 itertools.combinations(non_empty_classes, size_left_superclass)
#                 for size_left_superclass in range(1, number_non_empty_classes // 2 + 1)):
#             set_left_classes = set(left_classes)
#             set_right_classes = non_empty_classes - set_left_classes
#             if not set_left_classes or not set_right_classes:
#                 # A valid split must have at least one sample in each side
#                 continue
#             yield set_left_classes, set_right_classes

#     @staticmethod
#     def _get_twoing_contingency_table(contingency_table, values_num_samples, set_left_classes,
#                                       set_right_classes):
#         twoing_contingency_table = np.zeros((contingency_table.shape[0], 2), dtype=float)
#         superclass_index_num_samples = [0, 0]
#         for value, value_num_samples in enumerate(values_num_samples):
#             if value_num_samples == 0:
#                 continue
#             for class_index in set_left_classes:
#                 superclass_index_num_samples[0] += contingency_table[value][class_index]
#                 twoing_contingency_table[value][0] += contingency_table[value][class_index]
#             for class_index in set_right_classes:
#                 superclass_index_num_samples[1] += contingency_table[value][class_index]
#                 twoing_contingency_table[value][1] += contingency_table[value][class_index]
#         return twoing_contingency_table, superclass_index_num_samples

#     @staticmethod
#     def _two_class_trick(original_gini, class_index_num_samples, values_seen, values_num_samples,
#                          contingency_table, num_total_valid_samples):
#         def _calculate_value_class_ratio(values_seen, values_num_samples, contingency_table,
#                                          non_empty_class_indices):
#             # TESTED!
#             value_number_ratio = [] # [(value, number_on_second_class, ratio_on_second_class)]
#             second_class_index = non_empty_class_indices[1]
#             for curr_value in values_seen:
#                 number_second_non_empty = contingency_table[curr_value][second_class_index]
#                 value_number_ratio.append((curr_value,
#                                            number_second_non_empty,
#                                            number_second_non_empty/values_num_samples[curr_value]))
#             value_number_ratio.sort(key=lambda tup: tup[2])
#             return value_number_ratio

#         # We only need to sort values by the percentage of samples in second non-empty class with
#         # this value. The best split will be given by choosing an index to split this list of
#         # values in two.
#         (first_non_empty_class,
#          second_non_empty_class) = _get_non_empty_class_indices(class_index_num_samples)
#         if first_non_empty_class is None or second_non_empty_class is None:
#             return (float('-inf'), {0}, set())

#         value_number_ratio = _calculate_value_class_ratio(values_seen,
#                                                           values_num_samples,
#                                                           contingency_table,
#                                                           (first_non_empty_class,
#                                                            second_non_empty_class))

#         best_split_total_gini_gain = float('-inf')
#         best_last_left_index = 0

#         num_left_first = 0
#         num_left_second = 0
#         num_left_samples = 0
#         num_right_first = class_index_num_samples[first_non_empty_class]
#         num_right_second = class_index_num_samples[second_non_empty_class]
#         num_right_samples = num_total_valid_samples

#         for last_left_index, (last_left_value, last_left_num_second, _) in enumerate(
#                 value_number_ratio[:-1]):
#             num_samples_last_left_value = values_num_samples[last_left_value]
#             # num_samples_last_left_value > 0 always, since the values without samples were not
#             # added to the values_seen when created by cls._generate_value_to_index

#             last_left_num_first = num_samples_last_left_value - last_left_num_second

#             num_left_samples += num_samples_last_left_value
#             num_left_first += last_left_num_first
#             num_left_second += last_left_num_second
#             num_right_samples -= num_samples_last_left_value
#             num_right_first -= last_left_num_first
#             num_right_second -= last_left_num_second

#             curr_children_gini_index = _calculate_children_gini_index(num_left_first,
#                                                                       num_left_second,
#                                                                       num_right_first,
#                                                                       num_right_second,
#                                                                       num_left_samples,
#                                                                       num_right_samples)
#             curr_gini_gain = original_gini - curr_children_gini_index
#             if curr_gini_gain > best_split_total_gini_gain:
#                 best_split_total_gini_gain = curr_gini_gain
#                 best_last_left_index = last_left_index

#         # Let's get the values and split the indices corresponding to the best split found.
#         set_left_values = set([tup[0] for tup in value_number_ratio[:best_last_left_index + 1]])
#         set_right_values = set(values_seen) - set_left_values

#         return (best_split_total_gini_gain, set_left_values, set_right_values)
