#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Module containing class that generates a random attribute."""

import numpy as np


class RandomAttributeGenerator(object):
    """Generator of random attribute as in Coppersmith.

    Entries in contingency table are integers i.i.d. uniformily in [0, 7].
    """

    def __init__(self, seed):
        if seed:
            np.random.seed(seed)

    def generate(self, num_values, num_classes):
        """Returns random attribute with the given number of values and classes.
        """
        return np.random.randint(0, 8, (num_values, num_classes))
