#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Module containing class that saves experiment results info."""

import collections

ExperimentInfo = collections.namedtuple(
    "ExperimentInfo",
    ["impurity", "num_iterations", "superclasses_largest_frequence", "largest_class_frequency"])
