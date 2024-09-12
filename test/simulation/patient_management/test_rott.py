from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from simulation.patient_management.rott import RemovalOtherThanTreatment


def test_initialization():
    """Test Initialisation."""
    rott = RemovalOtherThanTreatment(horizon=30, seed=42)
    assert rott.horizon == 30
    assert rott.seed == 42
    assert rott.mean is None
    assert rott.std_dev is None


def test_setup_stochastic_distribution():
    """Test setup_stochastic_distribution."""
    rott = RemovalOtherThanTreatment(horizon=30)
    rott.setup_stochastic_distribution(mean=50.0, std_dev=10.0)

    assert rott.mean == 50.0
    assert rott.std_dev == 10.0


def test_setup_sql_distribution():
    """Test setup_sql_distribution with mocked SQL engine."""
    rott = RemovalOtherThanTreatment(horizon=30)

    mock_engine = MagicMock()
    query_string = "SELECT rott FROM daily_removals"

    mock_dataframe = pd.DataFrame({0: [1, 2, 0, 0, 4]})
    pd.read_sql = MagicMock(return_value=mock_dataframe)

    rott.setup_sql_distribution(mock_engine, query_string)

    assert rott.mean == mock_dataframe[0].mean()
    assert rott.std_dev == mock_dataframe[0].std()


def test_sql_distribution_error():
    """Test error if SQL returns more than one column."""
    rott = RemovalOtherThanTreatment(horizon=30)

    mock_engine = MagicMock()
    query_string = "SELECT rott, other_column FROM daily_removals"

    mock_dataframe = pd.DataFrame({"rott": [30, 40, 50], "other_column": [1, 2, 3]})
    pd.read_sql = MagicMock(return_value=mock_dataframe)

    with pytest.raises(ValueError, match="The DataFrame has more than one column."):
        rott.setup_sql_distribution(mock_engine, query_string)


def test_return_number_of_removals():
    """Test return_number_of_removals."""
    rott = RemovalOtherThanTreatment(horizon=10, seed=42)
    rott.setup_stochastic_distribution(mean=50.0, std_dev=10.0)

    removals = rott.return_number_of_removals()

    assert len(removals) == 10
    assert isinstance(removals, np.ndarray)
    assert np.all(removals >= 0)


def test_reproducibility():
    """Test reproducibility with seed."""
    rott1 = RemovalOtherThanTreatment(horizon=10, seed=42)
    rott1.setup_stochastic_distribution(mean=50.0, std_dev=10.0)
    removals1 = rott1.return_number_of_removals()

    rott2 = RemovalOtherThanTreatment(horizon=10, seed=42)
    rott2.setup_stochastic_distribution(mean=50.0, std_dev=10.0)
    removals2 = rott2.return_number_of_removals()

    assert np.array_equal(
        removals1, removals2
    ), "Results should be reproducible with the same seed."
