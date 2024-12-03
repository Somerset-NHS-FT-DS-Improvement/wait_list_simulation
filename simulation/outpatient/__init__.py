from typing import Optional

import numpy as np
import pandas as pd
import sqlalchemy as sa

from .. import parameterise_simulation
from ..patient_management.forecast_arrivals import Forecaster
from ..patient_management.patient_categoriser import patient_categoriser
from .resource_match import OutpatientResourceMatcher


# TODO: This code is almost completely duplicated from mri, should probably be using inheritance here...
class NewOutpatients:
    def __init__(
        self,
        historic_num_refs: pd.Series,
        historic_data: pd.DataFrame,
        forecast_horizon: int,
        new_patients_seed: Optional[int] = None,
        patient_categoriser_seed: Optional[int] = None,
    ) -> None:
        """
        Initialise the NewOutpatients class.

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

        self.category_columns = ["priority"]
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
            keys = keys if isinstance(keys, list) else [keys]
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
            df = self.historic_data.iloc[np.concatenate(patient_indices)].copy()
            df.loc[:, "days waited"] = 0

        return df


def parameterise_new_patient_object(
    engine: sa.engine.Engine,
    forecast_horizon: int,
    treatment_function_code: int,
    new_patient_seed: int = None,
    patient_categoriser_seed: int = None,
    site: str = "mph",
) -> "NewOutpatients":
    """
    Creates and returns an instance of the NewOutpatients object.

    Args:
        engine (sa.engine.Engine): Database engine to connect to the SQL database.
        forecast_horizon (int): The number of periods (e.g., days, weeks) over which the forecast is made.

        new_patient_seed (int, optional): Seed for random generation of new patients. Defaults to None.
        patient_categoriser_seed (int, optional): Seed for categorizing new patients. Defaults to None.

    Returns:
        NewOutpatients: An instance of NewOutpatients class initialized with required data.
    """

    mc = NewOutpatients(
        pd.read_sql(
            f"Execute [wl].[opa_admissions] @site={site}, @tfc='{treatment_function_code}'",
            engine,
        ),
        pd.read_sql(
            f"Execute [wl].[current_opa_waiting_list] @site={site}, @tfc='{treatment_function_code}'",
            engine,
        ),
        # TODO: Update this!
        # pd.read_sql(
        #     open(f"{path_to_sql_files}/MRI_historic_waiting_list.sql", "r").read(),
        #     engine,
        # ),
        forecast_horizon,
        new_patients_seed=new_patient_seed,
        patient_categoriser_seed=patient_categoriser_seed,
    )
    return mc


def get_initial_waiting_list(
    engine: sa.engine.Engine, treatment_function_code: int, site: str = None
) -> pd.DataFrame:
    """
    Retrieves the initial OP waiting list from the database.

    Args:
        engine (sa.engine.Engine): Database engine to connect to the SQL database.

    Returns:
        pd.DataFrame: A DataFrame containing the initial OP waiting list.
    """
    return pd.read_sql(
        f"Execute [wl].[current_opa_waiting_list] @site={site}, @tfc='{treatment_function_code}'",
        engine,
    )


def resource_dummy_function(in_list):
    num_patients_to_remove = np.random.randint(0, 10)
    return in_list.iloc[:num_patients_to_remove].index


def setup_op_simulation(
    path_to_sql_files: str, treatment_function_code: int, seed: int = None
) -> tuple[int, "Simulation"]:
    """
    Sets up and initializes the OP simulation.

    Args:
        path_to_sql_files (str): Path to the SQL files and other necessary setup files.

    Returns:
        tuple[int, Simulation]: A tuple containing the seed used for the simulation and the initialized simulation object.
    """
    engine = sa.create_engine(open(f"{path_to_sql_files}/engine.txt", "r").read())

    seed = np.random.default_rng().integers(0, 2**32) if not seed else seed

    # seeds
    seeds = np.random.default_rng(seed).integers(0, 2**32, 8)

    # length_of_simulation
    forecast_horizon = 5

    # new patient
    mc = parameterise_new_patient_object(
        engine,
        forecast_horizon,
        treatment_function_code,
        new_patient_seed=seeds[0],
        patient_categoriser_seed=seeds[1],
    )
    new_patient_function = mc.generate_new_patients

    # initial waiting list
    initial_waiting_list = get_initial_waiting_list(engine, treatment_function_code)

    # resource matching
    oprm = OutpatientResourceMatcher(engine, 110)
    resource_matching_function = oprm.match_resource

    # priority order
    priority_order = [
        "A&E patients",
        "inpatients",
        "Breach",
        "Days waited",
        "Over minimum wait time",
        "Under maximum wait time",
    ]

    # dna_rate, cancellation_rate, emergency_rate, fu_rate
    dna_rate = 0
    cancellation_rate = 0
    emergency_rate = 0
    fu_rate = 0
    dna_rng = np.random.default_rng(seed=seeds[2])
    cancellation_rng = np.random.default_rng(seed=seeds[3])
    emergency_rng = np.random.default_rng(seed=seeds[4])
    fu_rng = np.random.default_rng(seed=seeds[5])
    rott_dist_params = {"mean": 1, "stddev": 1}

    sim = parameterise_simulation(
        initial_waiting_list,
        new_patient_function,
        resource_matching_function,
        priority_order,
        dna_rate,
        cancellation_rate,
        forecast_horizon,
        rott_dist_params=rott_dist_params,
        rott_seed=seeds[6],
        capacity_seed=seeds[7],
    )

    return seed, sim
