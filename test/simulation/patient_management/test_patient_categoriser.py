import pytest
import pandas as pd
import numpy as np
from collections import Counter
from simulation.patient_management.forecast_arrivals import Forecaster
from simulation.patient_management.patient_categoriser import patient_categoriser


# Sample input data for the tests
@pytest.fixture
def sample_historic_data():
    # Create a DataFrame with hierarchical columns 'priority' and 'procedure'
    data = {
        'priority': ['urgent', 'non-urgent', 'urgent', 'non-urgent', 'urgent'],
        'procedure': ['surgery', 'therapy', 'therapy', 'surgery', 'surgery'],
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_fcst_additions():
    # Simulate the forecast additions as a pandas Series
    date_rng = pd.date_range(start='2023-01-01', periods=5, freq='D')
    return pd.Series([3, 2, 4, 5, 3], index=date_rng)

def test_patient_categoriser_basic(sample_fcst_additions, sample_historic_data):
    # Test basic functionality of the patient_categoriser function
    hierarchical_cols = ['priority', 'procedure']

    # Call the function
    output_df = patient_categoriser(
        fcst_additions=sample_fcst_additions,
        hierarchical_cols=hierarchical_cols,
        historic_data=sample_historic_data,
        seed=42
    )

    # Check if output is a DataFrame
    assert isinstance(output_df, pd.DataFrame), "Output should be a pandas DataFrame"

    # Check if 'cat' column exists in the output DataFrame
    assert 'cat' in output_df.columns, "'cat' column should exist in the output DataFrame"

    # Check if the output index matches the forecast addition dates
    assert all(output_df.index == sample_fcst_additions.index), "Index should match forecast addition dates"

def test_patient_categoriser_with_missing_data(sample_fcst_additions, sample_historic_data):
    # Test the function when historic data contains NaN values in hierarchical columns
    hierarchical_cols = ['priority', 'procedure']

    # Introduce some NaN values in the historic data
    historic_data_with_nan = sample_historic_data.copy()
    historic_data_with_nan.loc[0, 'procedure'] = np.nan

    # Call the function
    output_df = patient_categoriser(
        fcst_additions=sample_fcst_additions,
        hierarchical_cols=hierarchical_cols,
        historic_data=historic_data_with_nan,
        seed=42
    )

    # Check if output DataFrame is created correctly after dropping NaN values
    assert isinstance(output_df, pd.DataFrame), "Output should be a pandas DataFrame even with NaN values in historic data"
    assert len(output_df) == len(sample_fcst_additions), "Output length should match the forecast addition length"

def test_patient_categoriser_probabilities(sample_historic_data):
    # Test if probabilities are computed correctly based on the historic data
    hierarchical_cols = ['priority', 'procedure']

    # Call the function with a small forecast to verify probabilities
    fcst_additions = pd.Series([10], index=pd.date_range(start='2023-01-01', periods=1))

    output_df = patient_categoriser(
        fcst_additions=fcst_additions,
        hierarchical_cols=hierarchical_cols,
        historic_data=sample_historic_data,
        seed=42
    )

    # Extract the categories
    category_distribution = output_df['cat'].iloc[0]

    # Check that the categories were assigned based on the proportions in the historic data
    assert sum(category_distribution.values()) == 10, "The total count should match the forecast additions"

def test_patient_categoriser_random_seed_consistency(sample_fcst_additions, sample_historic_data):
    # Test that setting a random seed results in consistent outputs
    hierarchical_cols = ['priority', 'procedure']

    # Run the function twice with the same seed
    output_df_1 = patient_categoriser(
        fcst_additions=sample_fcst_additions,
        hierarchical_cols=hierarchical_cols,
        historic_data=sample_historic_data,
        seed=42
    )
    output_df_2 = patient_categoriser(
        fcst_additions=sample_fcst_additions,
        hierarchical_cols=hierarchical_cols,
        historic_data=sample_historic_data,
        seed=42
    )

    # Check that both outputs are identical
    pd.testing.assert_frame_equal(output_df_1, output_df_2, check_like=True)

