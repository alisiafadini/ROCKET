from rocket.llg.utils import newton_step
from numpy.testing import assert_almost_equal
import torch
import numpy as np


def test_newton_optimizer():
    example_function = lambda x: torch.tensor(x**2 + x + 1.0)
    example_derivative = lambda x: torch.tensor([2 * x + 1.0])
    example_hessian = lambda x: torch.tensor([[2.0]])

    expected_extremum = 0.75
    expected_x_at_extremum = -0.5
    x = 0.0

    delta = np.inf
    computed_extremum = np.inf

    while delta > 1e-7:
        old_estimate = computed_extremum
        x = newton_step(x, example_derivative(x), example_hessian(x))
        computed_extremum = example_function(x)
        delta = np.abs(computed_extremum - old_estimate)

    assert_almost_equal(computed_extremum, expected_extremum)
    assert_almost_equal(x, expected_x_at_extremum)


# TO DO remove this and use pytest
if __name__ == "__main__":
    test_newton_optimizer()
