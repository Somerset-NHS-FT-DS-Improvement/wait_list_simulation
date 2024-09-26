from typing import Optional

import numpy as np
import pandas as pd

from simulation.patient_management.forecast_arrivals import Forecaster
from simulation.patient_management.patient_categoriser import \
    patient_categoriser


class MRINewPatients:
    def __init__(
        self,
        historic_num_refs: pd.Series,
        historic_data: pd.DataFrame,
        forecast_horizon: int,
        new_patients_seed: Optional[int] = None,
        patient_categoriser_seed: Optional[int] = None,
    ) -> None:
        """
        Initialise the MRINewPatients class.

        Args:
            historic_num_refs (pd.Series): A series containing historical number of referrals.
            historic_data (pd.DataFrame): A dataframe containing historical data of patients.
            forecast_horizon (int): Number of days to forecast ahead.
            new_patients_seed (Optional[int]): Seed for the random number generator for patient generation. Default is None.
            patient_categoriser_seed (Optional[int]): Seed for the patient categoriser function. Default is None.
        """
        self.forecast_horizon = forecast_horizon
        self.historic_data = historic_data

        self.rng = np.random.default_rng(seed=new_patients_seed)

        self.category_columns = ["priority", "emerg_elec"]
        self.category_ratios = self.__generate_categories(
            historic_num_refs, historic_data, forecast_horizon, patient_categoriser_seed
        )

    def __generate_categories(
        self,
        historic_num_refs: pd.Series,
        historic_data: pd.DataFrame,
        forecast_horizon: int,
        seed: Optional[int],
    ) -> pd.DataFrame:
        """
        Generate category ratios for patient distribution using historical data and forecasting.

        Args:
            historic_num_refs (pd.Series): A series containing historical number of referrals.
            historic_data (pd.DataFrame): A dataframe containing historical data of patients.
            forecast_horizon (int): Number of days to forecast ahead.
            seed (Optional[int]): Seed for the patient categoriser function. Default is None.

        Returns:
            pd.DataFrame: A dataframe containing forecasted category ratios for each day.
        """
        fc = Forecaster(historic_num_refs, forecast_horizon)
        fc.forecast()
        fc.convert_to_count()

        return patient_categoriser(
            fc.forecast_data, self.category_columns, historic_data, seed
        )

    def generate_new_patients(self, day_number: int) -> pd.DataFrame:
        """
        Generate new patients for a given day number based on forecasted category ratios.

        Args:
            day_number (int): The day number for which new patients should be generated.

        Returns:
            pd.DataFrame: A dataframe containing the data of newly generated patients for the given day.

        Raises:
            IndexError: If the day_number exceeds the forecast horizon.
        """
        if day_number > self.forecast_horizon:
            raise IndexError(
                f"The day number of {day_number} exceeds the original forecast horizon of {self.forecast_horizon}."
            )

        ratios = self.category_ratios.iloc[day_number]["cat"]
        patient_indices = []
        for keys, val in ratios.items():
            matching_vals = np.logical_and.reduce(
                [
                    (self.historic_data[col] == key)
                    for col, key in zip(self.category_columns, keys)
                ]
            )
            patient_indices.append(
                self.rng.choice(self.historic_data[matching_vals].index, val)
            )

        if len(patient_indices) == 0:
            df = pd.DataFrame(columns=self.historic_data.columns)
        else:
            df = self.historic_data.iloc[np.concatenate(patient_indices)]
            df["wait_in_days"] = 0

        return df
