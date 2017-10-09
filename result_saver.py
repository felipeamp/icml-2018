#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Module containing class that saves experiment results to a csv."""

import collections
import os


def write_header(criteria, fout):
    """Writes csv header containing iterarion number and criteria names."""
    line_list = []
    line_list.append('iteration number')
    for criterion_name in criteria:
        line_list.append(criterion_name)
    print(','.join(line_list), file=fout)


class ResultSaver(object):
    """Stores impurity result for each criteria and saves in CSV."""
    def __init__(self, csv_output_path):
        if os.path.exists(csv_output_path):
            raise ValueError("csv_output_path (%s) must be a non-existent file." % csv_output_path)
        if not os.path.exists(os.path.dirname(csv_output_path)):
            raise ValueError("The directory to contain the output csv (%s) must exist." %
                             os.path.dirname(csv_output_path))
        self.csv_output_path = csv_output_path
        self.results = collections.defaultdict(list)

    def store_result(self, criterion_name, best_impurity_found):
        """Stores given result. Will be saved in csv later, when write_csv is called."""
        self.results[criterion_name].append(best_impurity_found)

    def write_csv(self):
        """Saves all results in csv given by self.csv_output_path."""
        criteria = sorted(self.results)
        max_iteration = max(len(criterion_results) for criterion_results in self.results.values())
        with open(self.csv_output_path, 'w') as fout:
            write_header(criteria, fout)
            for iteration in range(max_iteration):
                line_list = []
                line_list.append(iteration + 1)
                for criterion_name in criteria:
                    if len(self.results[criterion_name]) <= iteration:
                        line_list.append("")
                    else:
                        line_list.append(self.results[criterion_name][iteration])
                print(','.join(line_list), file=fout)
