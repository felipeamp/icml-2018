#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Module containing TreeNode class."""

import numpy as np


class ContingencyTable(object):
    """Holds contingency table and count of samples per value for an attribute.
    """
    def __init__(self, contingency_table, num_samples_per_value):
        self.contingency_table = contingency_table
        self.num_samples_per_value = num_samples_per_value


class TreeNode(object):
    """Contains information of a certain node of a decision tree.

        It has information about the samples used during training at this node and also about it's
    best split found.

    Attributes:
        split_info (SplitInfo): Data structure containing information about which attribute and
            split values were obtained in this TreeNode with a certain criterion. Also contains the
            criterion value. It is None if the current TreeNode is a leaf.
        contingency_tables (:obj:'list' of 'tuple' of 'list' of 'np.array'): contains a list where
            the i-th entry is a tuple containing two pieces of information of the i-th attribute:
            the contingency table for that attribute (value index is row, class index is column) and
            a list of number of times each value is attained in the training set (i-th entry is the
            number of times a sample has value i in this attribute and training dataset). Used by
            many criteria when calculating the optimal split. Note that, for invalid attributes, the
            entry is a tuple with empty lists ([], []).
        curr_dataset (Dataset): dataset containing the training samples.
        valid_nominal_attribute (:obj:'list' of 'bool'): list where the i-th entry indicates wether
            the i-th attribute from the dataset is valid and nominal or not.
        num_valid_samples (int): number of training samples in this TreeNode.
    """
    def __init__(self, curr_dataset, valid_nominal_attribute):
        """Initializes a TreeNode instance with the given arguments.

        Args:
            curr_dataset (Dataset): dataset of samples used for training/split generation.
            valid_nominal_attribute (:obj:'list' of 'bool'): the i-th entry informs whether the i-th
                attribute is a valid nominal one.
        """

        self.dataset = curr_dataset
        self.valid_nominal_attribute = valid_nominal_attribute
        self.all_best_splits = None
        self.contingency_tables = self._calculate_contingency_tables()

    def _calculate_contingency_tables(self):
        contingency_tables = [] # list of ContingencyTable
        for (attrib_index,
             is_valid_nominal_attribute) in enumerate(self.valid_nominal_attribute):
            if not is_valid_nominal_attribute:
                contingency_tables.append([])
                continue
            attrib_num_values = len(self.dataset.attrib_int_to_value[attrib_index])
            curr_contingency_table = np.zeros((attrib_num_values, self.dataset.num_classes),
                                              dtype=int)
            curr_num_samples_per_value = np.zeros((attrib_num_values), dtype=int)
            for sample_index in range(self.dataset.num_samples):
                curr_sample_value = self.dataset.samples[sample_index][attrib_index]
                curr_sample_class = self.dataset.sample_class[sample_index]
                curr_contingency_table[curr_sample_value][curr_sample_class] += 1
                curr_num_samples_per_value[curr_sample_value] += 1
            contingency_tables.append(ContingencyTable(
                curr_contingency_table, curr_num_samples_per_value))
        return contingency_tables

    def calculate_all_splits(self, criterion):
        """Splits the current TreeNode using the given splitting criterion.

        Args:
            criterion (Criterion): splitting criterion used to split the node.
        """
        self.all_best_splits = criterion.find_all_best_splits(self)
