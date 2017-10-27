#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Module containing class that saves monte carlo experiments results to a csv."""

import collections
import os

# import criteria


def write_csv_experiments_header(fout):
    """Writes csv_experiments header."""
    line_list = ["num_values", "num_classes", "experiment number", "method", "best impurity found",
                 "# iterations until convergence", "superclasses_largest_frequence",
                 "largest_class_frequency"]
    print(','.join(line_list), file=fout)


def write_csv_table_header(fout, all_criteria):
    """Writes csv_table header."""
    line_list = ["num_values", "num_classes", "experiment number"] + all_criteria
    print(','.join(line_list), file=fout)


class MonteCarloResultSaver(object):
    """Stores impurity result for each criteria and saves in CSV."""
    def __init__(self, csv_experiments_filename, csv_table_filename, csv_output_dir,
                 should_skip_experiment_fn):
        if csv_output_dir and not os.path.exists(csv_output_dir):
            raise ValueError("The directory to contain the output csv (%s) must exist." %
                             csv_output_dir)
        if not csv_output_dir:
            csv_output_dir = os.getcwd()
        csv_experiments_output_path = os.path.join(csv_output_dir, csv_experiments_filename)
        csv_table_output_path = os.path.join(csv_output_dir, csv_table_filename)
        if os.path.exists(csv_experiments_output_path):
            raise ValueError("csv_experiments (%s) must be a non-existent file." %
                             csv_experiments_output_path)
        if os.path.exists(csv_table_output_path):
            raise ValueError("csv_table (%s) must be a non-existent file." % csv_table_output_path)
        self.csv_experiments_output_path = csv_experiments_output_path
        self.csv_table_output_path = csv_table_output_path
        criterion_results_ctor = lambda: collections.defaultdict(list)
        # self.results[(num_values, num_classes)][criterion_name] = experiment_info
        self.results = collections.defaultdict(criterion_results_ctor)
        # self.should_skip_experiment_fn(num_values, num_classes, criterion_name) is bool
        self.should_skip_experiment_fn = should_skip_experiment_fn

    def store_result(self, num_values, num_classes, criterion_name, curr_experiment_info):
        """Stores given result. Will be saved in csv later, when write_csv is called."""
        self.results[(num_values, num_classes)][criterion_name].append(curr_experiment_info)
        # DEBUG
        # curr_is_twoing = False
        # other_name = None
        # if criterion_name == criteria.RANDOM_CLASS_PARTITION_K_CLASS_GINI.name:
        #     other_name = criteria.TWOING_K_CLASS_GINI.name
        # elif criterion_name == criteria.TWOING_K_CLASS_GINI.name:
        #     other_name = criteria.RANDOM_CLASS_PARTITION_K_CLASS_GINI.name
        #     curr_is_twoing = True
        # elif criterion_name == criteria.RANDOM_CLASS_PARTITION_K_CLASS_ENTROPY.name:
        #     other_name = criteria.TWOING_K_CLASS_ENTROPY.name
        # elif criterion_name == criteria.TWOING_K_CLASS_ENTROPY.name:
        #     other_name = criteria.RANDOM_CLASS_PARTITION_K_CLASS_ENTROPY.name
        #     curr_is_twoing = True
        # if (other_name in self.results[(num_values, num_classes)] and
        #     (len(self.results[(num_values, num_classes)][other_name]) ==
        #      len(self.results[(num_values, num_classes)][criterion_name]))):
        #     other_impurity = self.results[(num_values, num_classes)][other_name][-1][0]
        #     if curr_is_twoing and other_impurity < best_impurity_found:
        #         print("FOUND PROBLEM!")
        #         print("experiment number:",
        #               len(self.results[(num_values, num_classes)][criterion_name]) - 1)
        #         print("twoing impurity:", best_impurity_found)
        #         print("random_class_partition impurity:", other_impurity)
        #     elif not curr_is_twoing and other_impurity > best_impurity_found:
        #         print("FOUND PROBLEM!")
        #         print("experiment number:",
        #               len(self.results[(num_values, num_classes)][criterion_name]) - 1)
        #         print("twoing impurity:", other_impurity)
        #         print("random_class_partition impurity:", best_impurity_found)

    def _get_criteria_correct_order(self, pairs_num_values_classes):
        unordered_criteria = []
        for pair_num_values_classes in pairs_num_values_classes:
            if len(self.results[pair_num_values_classes]) > len(unordered_criteria):
                unordered_criteria = self.results[pair_num_values_classes]
        gini_criteria = []
        entropy_criteria = []
        unkown = []
        for criterion in unordered_criteria:
            if "Gini" in criterion:
                gini_criteria.append(criterion)
            elif "Entropy" in criterion:
                entropy_criteria.append(criterion)
            else:
                unkown.append(criterion)
        all_criteria = sorted(gini_criteria) + sorted(unkown) + sorted(entropy_criteria)
        return all_criteria

    def _save_csv_experiments(self, pair_num_values_classes, num_experiments, all_criteria):
        with open(self.csv_experiments_output_path, 'w') as fout:
            write_csv_experiments_header(fout)
            for experiment_num in range(num_experiments):
                for (num_values, num_classes) in pair_num_values_classes:
                    for criterion_name in all_criteria:
                        if self.should_skip_experiment_fn(num_values, num_classes, criterion_name):
                            continue
                        curr_experiment_info = self.results[
                            (num_values, num_classes)][criterion_name][experiment_num]
                        line_list = map(str,
                                        [num_values, num_classes, experiment_num + 1,
                                         criterion_name, curr_experiment_info.impurity,
                                         curr_experiment_info.num_iterations,
                                         curr_experiment_info.superclasses_largest_frequence,
                                         curr_experiment_info.largest_class_frequency])
                        print(','.join(line_list), file=fout)

    def _save_csv_table(self, pair_num_values_classes, num_experiments, all_criteria):
        with open(self.csv_table_output_path, 'w') as fout:
            write_csv_table_header(fout, all_criteria)
            for (num_values, num_classes) in pair_num_values_classes:
                for experiment_num in range(num_experiments):
                    line_list = [num_values, num_classes, experiment_num + 1]
                    for criterion_name in all_criteria:
                        if self.should_skip_experiment_fn(num_values, num_classes, criterion_name):
                            continue
                        curr_experiment_info = self.results[
                            (num_values, num_classes)][criterion_name][experiment_num]
                        line_list.append(curr_experiment_info.impurity)
                    print(','.join(map(str, line_list)), file=fout)

    def write_csv(self):
        """Saves aggregated experiments results in csv given by self.csv_output_path."""
        pairs_num_values_classes = sorted(self.results)
        all_criteria = self._get_criteria_correct_order(pairs_num_values_classes)
        num_experiments = max(
            len(self.results[pairs_num_values_classes[0]][criterion])
            for criterion in all_criteria)
        # # DEBUG
        # print("pairs_num_values_classes:", pairs_num_values_classes)
        # print("all_criteria:", all_criteria)
        # print("num_experiments:", num_experiments)
        try:
            self._save_csv_experiments(pairs_num_values_classes, num_experiments, all_criteria)
        except Exception as err:
            # DEBUG
            print("EXCEPT IN EXPERIMENT WRITING:", err)
        try:
            self._save_csv_table(pairs_num_values_classes, num_experiments, all_criteria)
        except Exception as err:
            # DEBUG
            print("EXCEPT IN TABLE WRITING:", err)
