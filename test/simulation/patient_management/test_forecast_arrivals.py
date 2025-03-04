import numpy as np
import pandas as pd
import pytest
from holidays import UK
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA

from simulation.patient_management.forecast_arrivals import Forecaster


# Sample test data
@pytest.fixture
def sample_data():
    # Create a DataFrame with daily 'dst' and 'refs' columns
    date_rng = pd.date_range(start="2022-01-01", end="2023-01-10", freq="D")
    df = pd.DataFrame(date_rng, columns=["dst"])
    df["refs"] = np.random.randint(1, 10, size=(len(date_rng)))
    return df


@pytest.fixture
def forecaster_instance(sample_data):
    # Create an instance of Forecaster with forecast horizon of 14 days
    return Forecaster(sample_data, fh=14)


def test_pre_processing(forecaster_instance):
    # Test pre-processing method
    forecaster_instance._pre_processing()

    # Check if data is structured correctly
    assert isinstance(
        forecaster_instance.data, pd.Series
    ), "Data should be a pandas Series"
    assert (
        not forecaster_instance.data.isnull().values.any()
    ), "There should be no missing values after pre-processing"


def test_create_holiday_dataframe(forecaster_instance):
    # Test holiday dataframe creation

    forecaster_instance._pre_processing()  # Must preprocess data first
    fcst_range = pd.date_range(
        start="2022-01-01", periods=len(forecaster_instance.data) + 14, freq="D"
    )
    forecaster_instance._create_holiday_dataframe(date_range=fcst_range)

    holidays = forecaster_instance.holidays
    assert isinstance(holidays, pd.DataFrame), "Holidays should be a pandas DataFrame"
    assert (
        not holidays.isnull().values.any()
    ), "Holidays dataframe should not have any NaNs"
    assert holidays.index.name == "dst", "Holiday DataFrame index should be named 'dst'"


def test_forecast(forecaster_instance):
    # Test forecast method
    forecaster_instance.forecast()

    assert isinstance(
        forecaster_instance.forecast_data, pd.DataFrame
    ), "Forecast output should be a DataFrame"
    assert (
        "yhat" in forecaster_instance.forecast_data.columns
    ), "Forecast output should contain 'yhat' column"
    assert (
        len(forecaster_instance.forecast_data) == forecaster_instance.fh
    ), "Forecast horizon should match the input fh"


def test_apply_growth(forecaster_instance):
    # Test applying growth
    forecaster_instance.forecast()
    forecaster_instance.apply_growth(annual_growth_rate=0.05)

    grown_forecast = forecaster_instance.forecast_data

    assert isinstance(
        grown_forecast, pd.DataFrame
    ), "Forecast with growth should be a DataFrame"
    assert (
        (grown_forecast >= forecaster_instance.forecast_data).all().all()
    ), "Forecast with growth should not decrease any values"


def test_convert_to_count(forecaster_instance):
    # Test conversion to count data
    forecaster_instance.forecast()

    forecaster_instance.convert_to_count()
    count_forecast = forecaster_instance.forecast_data

    assert isinstance(
        count_forecast, pd.DataFrame
    ), "Converted forecast should be a pandas DataFrame"
    assert count_forecast.yhat.apply(
        lambda x: isinstance(x, int)
    ).all(), "Converted forecast should contain integer values"
    assert (
        count_forecast.yhat >= 0
    ).all(), "Count data should not contain negative values"


def test_fit_error(forecaster_instance):
    # Test model error calculation
    forecaster_instance.forecast()

    residuals = forecaster_instance.fit_error()

    assert isinstance(residuals, pd.Series), "Residuals should be a pandas Series"
    assert len(residuals) == len(
        forecaster_instance.data
    ), "Residuals length should match input data length"
