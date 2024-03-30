"""
Unit Test Case based on Sample Solution 1
"""

import numpy as np
import pytest
import main as main

@pytest.fixture
def outcome():
    return np.loadtxt('test1.csv',
                 delimiter=",", dtype=int)

def test_overallocation(outcome):
    assert main.overallocation(outcome) == 37, 'overallocation objective ' \
                                      'is not properly calculated'

def test_conflicts(outcome):
    assert main.conflicts(outcome) == 8, 'conflicts objective ' \
                                           'is not properly calculated'

def test_undersupport(outcome):
    assert main.undersupport(outcome) == 1, 'undersupport objective ' \
                                        'is not properly calculated'

def test_unwilling(outcome):
    assert main.unwilling(outcome) == 53, 'unwilling objective ' \
                                        'is not properly calculated'

def test_unpreferred(outcome):
    assert main.unpreferred(outcome) == 15, 'unpreferred objective ' \
                                      'is not properly calculated'


