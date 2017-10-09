#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Module containing class that saves monte carlo experiments results to a csv."""

import collections
import os


def write_header(fout):
    """Writes csv header."""
    line_list = ["num_values", "num_classes", "experiment number", "method", "best impurity found",
                 "# iterations until convergence"]
    print(','.join(line_list), file=fout)


class MonteCarloResultSaver(object):
    """Stores impurity result for each criteria and saves in CSV."""
    def __init__(self, csv_output_path):
        if os.path.exists(csv_output_path):
            raise ValueError("csv_output_path (%s) must be a non-existent file." % csv_output_path)
        if not os.path.exists(os.path.dirname(csv_output_path)):
            raise ValueError("The directory to contain the output csv (%s) must exist." %
                             os.path.dirname(csv_output_path))
        self.csv_output_path = csv_output_path
        # self.results[(num_values, num_classes)][criterion_name] = (impurity, num_iterations)
        criterion_results_ctor = lambda: collections.defaultdict(list)
        self.results = collections.defaultdict(criterion_results_ctor)

    def store_result(self, num_values, num_classes, criterion_name, best_impurity_found,
                     num_iterations_when_converged):
        """Stores given result. Will be saved in csv later, when write_csv is called."""
        self.results[(num_values, num_classes)][criterion_name].append(
            (best_impurity_found, num_iterations_when_converged))

    def write_csv(self):
        """Saves aggregated experiments results in csv given by self.csv_output_path."""
        pair_num_values_classes = sorted(self.results)
        criteria = sorted(self.results[pair_num_values_classes[0]])
        num_experiments = len(self.results[pair_num_values_classes[0]][criteria[0]])
        with open(self.csv_output_path, 'w') as fout:
            write_header(fout)
            for experiment_num in range(num_experiments):
                for (num_values, num_classes) in pair_num_values_classes:
                    for criterion_name in criteria:
                        (best_impurity_found, num_iterations_when_converged) = self.results[
                            (num_values, num_classes)][criterion_name]
                        line_list = [num_values, num_classes, experiment_num + 1, criterion_name,
                                     best_impurity_found, num_iterations_when_converged]
                        print(','.join(line_list), file=fout)
