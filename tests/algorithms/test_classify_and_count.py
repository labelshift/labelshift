"""Classify and count algorithm."""
import numpy as np
import numpy.testing as nptest

from labelshift import algorithms


# TODO(pawel-czyz): This could be refactored into a fixture.
predictions = [
    [0.4, 0.6],
    [0.7, 0.3],
    [0.1, 0.9],
]


def test_list() -> None:
    """Test if works for a list."""
    prevalences = algorithms.classify_and_count(predictions)
    nptest.assert_allclose(prevalences, [1 / 3, 2 / 3])


def test_array() -> None:
    """Test if works for a numpy array."""
    prevalences = algorithms.classify_and_count(np.array(predictions))
    nptest.assert_allclose(prevalences, [1 / 3, 2 / 3])
