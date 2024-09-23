import holidays
import numpy as np
import pandas as pd
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA


class Forecaster:

    def __init__(self, data_in, fh):
        """
        Forecasting class for referrals.

        Args:
            data_in (pd.DataFrame): Assuming the data will be daily with columns dst and refs.
            fh (int): Output forecast horizon (in days).

        """

        self.data_in = data_in
        self.fh = fh

    def pre_processing(self):
        """
        Structures data in required format for sk-time framework.
        """

        input_data = self.data_in.set_index("dst").asfreq("d").fillna(0)
        self.data = input_data[["refs"]].rename(columns={"refs": "y"})["y"]

    def _create_holiday_dataframe(self):
        """
        Create daily binary dataframe for English holidays during historic/forecast period.
        To use as external regressors.
        """

        date_range = pd.date_range(
            start=self.data.index.min(), periods=len(self.data) + self.fh, freq="D"
        )

        # Create a dict-like object for England's public holidays
        uk_holidays = holidays.UK(
            state="England",
            years=range(date_range.min().year, date_range.max().year + 1),
        )

        holiday_vals = [
            "New Year's Day",
            "Christmas Day",
            "Good Friday",
            "Easter Monday",
            "May Day",
            "Spring Bank Holiday",
            "Late Summer Bank Holiday",
            "Boxing Day",
        ]

        uk_holidays = pd.DataFrame(uk_holidays.items(), columns=["ds", "holiday"])
        uk_holidays["ds"] = pd.to_datetime(uk_holidays["ds"])

        holiday_df = uk_holidays[uk_holidays.ds.isin(date_range)]
        holiday_df = holiday_df[holiday_df.holiday.isin(holiday_vals)]

        exog_holiday_df = pd.DataFrame()
        exog_holiday_df["ds"] = date_range

        exog_holiday_df["ds"] = pd.to_datetime(exog_holiday_df["ds"])
        holiday_df["ds"] = holiday_df["ds"].values

        for x in holiday_df.holiday.unique():
            exog_holiday_df[x] = (
                exog_holiday_df.ds.isin(holiday_df[holiday_df.holiday == x].ds) * 1
            )

        holidays_data = exog_holiday_df.set_index("ds", drop=True).sort_index()
        holidays_data.index.name = "dst"
        self.holidays = holidays_data.astype("float")

    def forecast(self, model=StatsForecastAutoARIMA(sp=7)):
        """
        Create forecast output.

        !! ToDo: Use approach specific to count data (GLM?)
        Args:
            - model (from sk-time)
        """

        self._create_holiday_dataframe()

        self.model = model
        self.model.fit(y=self.data, X=self.holidays.loc[self.data.index])

        fcst_range = pd.date_range(
            start=self.data.index.min(), periods=len(self.data) + self.fh, freq="D"
        )

        # Output forecast.
        forecast = self.model.predict(
            fh=np.arange(1, 1 + self.fh), X=self.holidays.loc[fcst_range]
        )

        self.forecast = forecast.to_frame(name="yhat")

    def apply_growth(self, annual_growth_rate=0):
        """
        Applies linear % annual growth to output forecast.

        Args:
            annual_growth_rate (int, 0-1): Annual growth %.
        Returns:
            Forecast with added growth.
        """

        self.annual_growth_rate = annual_growth_rate

        forecasts_df = self.forecast

        daily_growth_factor = [
            ((self.annual_growth_rate / 365) * i)
            for i in range(1, len(self.forecast) + 1)
        ]

        forecast_sums = forecasts_df.sum()

        growth_amounts = forecast_sums * self.annual_growth_rate
        growth = [
            growth_amounts.values[x] * np.array(daily_growth_factor)
            for x in range(0, len(forecast_sums.index))
        ]

        growth_addition = pd.DataFrame(growth).T

        growth_addition.columns = forecast_sums.index
        growth_addition.index = forecasts_df.index

        return forecasts_df.add(growth_addition)

    def convert_to_count(self, continuous_data):
        """
        Convert to count data.

        !! ToDo: Need to think of better approach.
        Args:
            continuous_data (Pandas.DataFrame/Series/list/array): Continuous data to convert to count.
        Returns:
            Count data.
        """

        self.continuous_data = continuous_data

        remove_zeros = np.maximum(0, self.continuous_data)

        count_forecast = np.round(remove_zeros, 0).astype(int)

        return count_forecast

    def fit_error(self):
        """
        Requires .forecast() to be called.

        Returns:
            Residual model error.
        """

        return self.model.predict_residuals(X=self.holidays)
