import holidays
import numpy as np
import pandas as pd
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA


class Forecaster:

    def __init__(self, data_in, fh, model=StatsForecastAutoARIMA(sp=7)):
        """
        A forecasting class for predicting referrals.

        Args:
            data_in (pd.DataFrame): Input data, expected to be a daily time series
                                    with at least two columns:
                                    - 'dst': Date or timestamp column.
                                    - 'refs': Referrals data.
            fh (int): Forecast horizon, representing the number of future periods (in days)
                      for which predictions will be generated.
            model (sktime object, optional): The forecasting model to be used. Defaults to
                                             StatsForecastAutoARIMA with a seasonal period (sp) of 7.

        Attributes:
            data (pd.DataFrame): The transformed data formatted in a structure suitable for the
                                 sktime model.
            forecast_data (pd.DataFrame): The output dataframe containing the forecasted results.
        """

        self.data_in = data_in
        self.fh = fh
        self.model = model
        self.data = None
        self.forecast_data = None

    def forecast(self):
        """
        Create forecast output.

        !! ToDo: Use approach specific to count data (GLM?)
        Args:
            - model (from sk-time)
        """

        self._pre_processing()

        fcst_range = pd.date_range(
            start=self.data.index.min(), periods=len(self.data) + self.fh, freq="D"
        )
        self._create_holiday_dataframe(date_range=fcst_range)

        self.model.fit(y=self.data, X=self.holidays.loc[self.data.index])

        # Output forecast.
        forecast = self.model.predict(
            fh=np.arange(1, 1 + self.fh), X=self.holidays.loc[fcst_range]
        )

        self.forecast_data = forecast.to_frame(name="yhat")

    def _pre_processing(self):
        """
        Structures data in required format for sk-time framework.
        """

        input_data = self.data_in.set_index("dst").asfreq("d").fillna(0)
        self.data = input_data[["refs"]].rename(columns={"refs": "y"})["y"]

    def _create_holiday_dataframe(self, date_range):
        """
        Create daily binary dataframe for English holidays during historic/forecast period.
        To use as external regressors.
        """

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
            exog_holiday_df[x] = exog_holiday_df.ds.isin(
                holiday_df[holiday_df.holiday == x].ds
            )

        holidays_data = exog_holiday_df.set_index("ds", drop=True).sort_index()
        holidays_data.index.name = "dst"
        self.holidays = holidays_data.astype(
            "float"
        )  # Convert to float (sktime requirement)

    def apply_growth(self, annual_growth_rate=0):
        """
        Applies linear % annual growth to output forecast.

        Args:
            annual_growth_rate (int, 0-1): Annual growth %.
        Returns:
            Forecast with added growth.
        """

        daily_growth_factor = [
            ((annual_growth_rate / 365) * i)
            for i in range(1, len(self.forecast_data) + 1)
        ]

        forecast_sums = self.forecast_data.sum()

        growth_amounts = forecast_sums * annual_growth_rate
        growth = [
            growth_amounts.values[x] * np.array(daily_growth_factor)
            for x in range(0, len(forecast_sums.index))
        ]

        growth_addition = pd.DataFrame(growth).T

        growth_addition.columns = forecast_sums.index
        growth_addition.index = self.forecast_data.index

        self.forecast_data.add(growth_addition)

    def convert_to_count(self):
        """
        Convert to count data.

        !! ToDo: Need to think of better approach.
        Args:
            continuous_data (Pandas.DataFrame/Series/list/array): Continuous data to convert to count.
        Returns:
            Count data.
        """

        remove_zeros = np.maximum(0, self.forecast_data)

        self.forecast_data = np.round(remove_zeros, 0).astype(int)

    def fit_error(self):
        """
        Requires .forecast() to be called.

        Returns:
            Residual model error.
        """

        return self.model.predict_residuals(X=self.holidays)
