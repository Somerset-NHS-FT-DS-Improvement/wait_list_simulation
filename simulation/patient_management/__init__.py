from typing import Optional

import numpy as np
import pandas as pd
import sqlalchemy as sa

from .forecast_arrivals import Forecaster
from .patient_categoriser import patient_categoriser


class Data:
    def __init__(
        self,
        path_to_sql_queries,
        historic_waiting_list_file_name,
        num_new_refs_file_name,
        current_waiting_list_file_name,
        fu_file_name=None,
        dna_file_name=None,
        cancellation_file_name=None,
        discharge_file_name=None,
        rott_file_name=None,
    ):
        self.engine = sa.create_engine(
            open(f"{path_to_sql_queries}/engine.txt", "r").read()
        )
        self.path_to_sql_queries = path_to_sql_queries

        self.fu_rate = None
        self.dna_rate = None
        self.cancellation_rate = None

        if fu_file_name is not None:
            self.fu_rate = self._run_sql(fu_file_name).values[0, 0]

        if dna_file_name is not None:
            self.dna_rate = self._run_sql(dna_file_name).values[0, 0]

        if cancellation_file_name is not None:
            self.cancellation_rate = self._run_sql(cancellation_file_name).values[0, 0]

        if discharge_file_name is not None:
            self.discharge_rage = self._run_sql(discharge_file_name).values[0, 0]

        if rott_file_name is not None:
            self.rott = self._run_sql(rott_file_name).iloc[0, :].to_dict()

        self.historic_waiting_list = self._run_sql(historic_waiting_list_file_name)

        self.num_new_refs = self._run_sql(num_new_refs_file_name)

        self.current_waiting_list = self._run_sql(current_waiting_list_file_name)

    def _run_sql(self, filename):
        return pd.read_sql(
            open(f"{self.path_to_sql_queries}/{filename}", "r").read(),
            self.engine,
        )


class NewPatients:
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

        if day_number % 15 == 0:
            print(f"{day_number} Reached")

        ratios = self.category_ratios.iloc[day_number]["cat"]

        patient_indices = [
            self.rng.choice(
                self.historic_data[
                    (self.historic_data[self.category_columns] == keys).all(axis=1)
                ].index,
                val,
            )
            for keys, val in ratios.items()
        ]

        if len(patient_indices) == 0:
            df = pd.DataFrame(columns=self.historic_data.columns)
        else:
            df = self.historic_data.iloc[np.concatenate(patient_indices)].copy()
            df.loc[:, "days waited"] = 0

        return df