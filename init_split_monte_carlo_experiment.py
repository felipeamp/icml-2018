#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Module that runs Monte Carlo experiments with different criterion using different inits."""

import argparse
import random

import numpy as np

import attribute_generator
import criteria
import dataset
import monte_carlo_result_saver
import tree_node



NUM_MONTE_CARLO_EXPERIMENTS = 10000

PAIR_NUM_VALUES_CLASSES = [(6, 3), (6, 9), (12, 3), (12, 9), (50, 3), (50, 9)]

CRITERIA_AND_SPLIT_IMPURITY_FN = [
    (criteria.RANDOM_FLIPFLOP_GINI, criteria.calculate_split_gini_index),
    (criteria.LARGEST_ALONE_FLIPFLOP_GINI, criteria.calculate_split_gini_index),
    (criteria.LIST_SCHEDULING_FLIPFLOP_GINI, criteria.calculate_split_gini_index),
    (criteria.RANDOM_CLASS_PARTITION_GINI, criteria.calculate_split_gini_index),
    (criteria.TWOING_GINI, criteria.calculate_split_gini_index),
    (criteria.GINI_GAIN, criteria.calculate_split_gini_index),
    (criteria.RANDOM_FLIPFLOP_ENTROPY, criteria.calculate_information_gain),
    (criteria.LIST_SCHEDULING_FLIPFLOP_ENTROPY, criteria.calculate_information_gain),
    (criteria.LARGEST_ALONE_FLIPFLOP_ENTROPY, criteria.calculate_information_gain),
    (criteria.RANDOM_CLASS_PARTITION_ENTROPY, criteria.calculate_information_gain),
    (criteria.TWOING_ENTROPY, criteria.calculate_information_gain),
    (criteria.INFORMATION_GAIN, criteria.calculate_information_gain)
    ]

SEED = 19880531


def remove_flipflop_from_name(criterion_name):
    return criterion_name.replace("-FlipFlop", "")


def run_experiment(curr_tree_node, criterion, split_impurity_fn, result_saver):
    """Runs the given experiment and saves it in the result_saver."""
    best_split = criterion.find_best_split_fn(tree_node=curr_tree_node, attrib_index=0)
    best_impurity_found = split_impurity_fn(
        curr_tree_node.dataset.num_samples,
        curr_tree_node.contingency_tables[0].contingency_table,
        curr_tree_node.contingency_tables[0].num_samples_per_value,
        best_split.left_values,
        best_split.right_values)
    num_values, num_classes = curr_tree_node.contingency_tables[0].contingency_table.shape
    result_saver.store_result(num_values, num_classes, remove_flipflop_from_name(criterion.name),
                              best_impurity_found, best_split.iteration_number)


def create_fake_tree_node(contingency_table):
    """Creates a fake TreeNode consistent with the given contingency table."""
    num_classes = contingency_table.shape[1]
    fake_dataset = dataset.Dataset(None, None, None, None, None, load_dataset=False)
    fake_dataset.num_classes = num_classes
    fake_dataset.num_samples = np.sum(contingency_table)
    fake_tree_node = tree_node.TreeNode(fake_dataset, [True], calculate_contingency_tables=False)
    num_samples_per_value = np.sum(contingency_table, axis=1)
    fake_tree_node.contingency_tables = [
        tree_node.ContingencyTable(contingency_table, num_samples_per_value)]
    return fake_tree_node


def main(csv_output_filepath):
    """Runs all experiments defined by the cartesian product of this module global variables."""
    criteria.MAX_ITERATIONS = 0
    result_saver = monte_carlo_result_saver.MonteCarloResultSaver(csv_output_filepath)
    attrib_gen = attribute_generator.RandomAttributeGenerator(SEED)
    random.seed(SEED)
    try:
        for (num_values, num_classes) in PAIR_NUM_VALUES_CLASSES:
            for _ in range(NUM_MONTE_CARLO_EXPERIMENTS):
                contingency_table = attrib_gen.generate(num_values, num_classes)
                curr_tree_node = create_fake_tree_node(contingency_table)
                for criterion, split_impurity_fn in CRITERIA_AND_SPLIT_IMPURITY_FN:
                    if (num_values > 12 and
                            (criterion.name == criteria.INFORMATION_GAIN.name or
                             criterion.name == criteria.GINI_GAIN.name)):
                        continue
                    run_experiment(curr_tree_node, criterion, split_impurity_fn, result_saver)
    finally:
        result_saver.write_csv()


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--csv_output_filepath',
                        help='Path to csv file to be created with the experiments results.')
    main(PARSER.parse_args().csv_output_filepath)
