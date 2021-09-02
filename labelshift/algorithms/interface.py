from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike


class QuantificationAlgorithm(Protocol):
    """Interface for quantification algorithms.

    Methods:
        quantify, applies quantification algorithm to the classifier's predictions
    """

    def quantify(
        *,
        test_predictions: ArrayLike,
        training_ground_truth: ArrayLike,
        training_predictions: ArrayLike,
    ) -> np.ndarray:
        """Quantifies test population.

        Args:


        Returns:

        """
        raise NotImplementedError
