from typing import Optional

import numpy as np
import pandas as pd
import sqlalchemy as sa

from .. import parameterise_simulation
from ..patient_management.forecast_arrivals import Forecaster
from ..patient_management.patient_categoriser import patient_categoriser
from .department import MRIDepartment
from .metrics import MRIMetrics


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

        if day_number % 15 == 0:
            print(f"{day_number} Reached")

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
            df = self.historic_data.iloc[np.concatenate(patient_indices)].copy()
            df.loc[:, "days waited"] = 0

        return df


def parameterise_new_patient_object(
    engine: sa.engine.Engine,
    forecast_horizon: int,
    path_to_sql_files: str,
    new_patient_seed: int = None,
    patient_categoriser_seed: int = None,
) -> "MRINewPatients":
    """
    Creates and returns an instance of the MRINewPatients object.

    Args:
        engine (sa.engine.Engine): Database engine to connect to the SQL database.
        forecast_horizon (int): The number of periods (e.g., days, weeks) over which the forecast is made.
        path_to_sql_files (str): Path to the SQL files for retrieving data.
        new_patient_seed (int, optional): Seed for random generation of new patients. Defaults to None.
        patient_categoriser_seed (int, optional): Seed for categorizing new patients. Defaults to None.

    Returns:
        MRINewPatients: An instance of MRINewPatients class initialized with required data.
    """
    # df = pd.read_sql(
    #     open(f"{path_to_sql_files}/MRI_historic_waiting_list.sql", "r").read(),
    #     engine,
    # )
    df = pd.read_csv(
        "S:\Data Science\sandboxes\chris\Waiting_list_sim\sample_historical_list.csv",
        index_col=0,
    )

    df["priority"] = df["priority"].str.strip()

    df["priority"] = df["priority"].fillna("Urgent")
    df["duration_mins"] = df["duration_mins_x"].fillna(30)

    mc = MRINewPatients(
        pd.read_sql(open(f"{path_to_sql_files}/num_new_refs.sql", "r").read(), engine),
        df,
        forecast_horizon,
        new_patients_seed=new_patient_seed,
        patient_categoriser_seed=patient_categoriser_seed,
    )

    return mc


def get_initial_waiting_list(
    engine: sa.engine.Engine, path_to_sql_files: str
) -> pd.DataFrame:
    """
    Retrieves the initial MRI waiting list from the database.

    Args:
        engine (sa.engine.Engine): Database engine to connect to the SQL database.
        path_to_sql_files (str): Path to the SQL files for retrieving data.

    Returns:
        pd.DataFrame: A DataFrame containing the initial MRI waiting list.
    """
    # df = pd.read_sql(
    #     open(f"{path_to_sql_files}/MRI_current_waiting_list.sql", "r").read(), engine
    # )
    df = pd.read_csv(
        "S:\Data Science\sandboxes\chris\Waiting_list_sim\sample_current_list.csv",
        index_col=0,
    )
    df["priority"] = df["priority"].str.strip()

    # These values taken from a meeting with the MRI dept
    df["priority"] = df["priority"].fillna("Urgent")
    df["duration_mins"] = df["duration_mins"].fillna(30)

    return df


def setup_mri_simulation(
    path_to_sql_files: str,
    dna_rate: float = None,
    cancellation_rate: float = None,
    fu_rate: float = None,
    clinic_utilisation: float = 1,
    seed: int = None,
) -> tuple[int, "Simulation"]:
    """
    Sets up and initializes the MRI simulation.

    Args:
        path_to_sql_files (str): Path to the SQL files and other necessary setup files.

    Returns:
        tuple[int, Simulation]: A tuple containing the seed used for the simulation and the initialized simulation object.
    """
    engine = sa.create_engine(open(f"{path_to_sql_files}/engine.txt", "r").read())

    seed = np.random.default_rng().integers(0, 2**32) if not seed else seed

    # seeds
    seeds = np.random.default_rng(seed).integers(0, 2**32, 8)

    new_patient_seed = (seeds[0],)
    patient_categoriser_seed = (seeds[1],)
    dna_seed = seeds[2]
    cancellation_seed = seeds[3]
    emergency_rng = np.random.default_rng(seed=seeds[4])
    fu_rng = np.random.default_rng(seed=seeds[5])
    rott_seed = seeds[6]
    capacity_seed = seeds[7]

    if dna_rate is None:
        dna_rate = pd.read_sql(
            open(f"{path_to_sql_files}/MRI_dna_rate.sql", "r").read(), engine
        ).values[0, 0]
    if cancellation_rate is None:
        cancellation_rate = pd.read_sql(
            open(f"{path_to_sql_files}/MRI_cancellation_rate.sql", "r").read(), engine
        ).values[0, 0]
    emergency_rate = 0
    if fu_rate is None:
        # TODO: put a SQL query here!
        fu_rate = 0
    # TODO: parameterise this from the sql
    rott_dist_params = {"mean": 0, "stddev": 0.0001}

    # length_of_simulation
    forecast_horizon = 365

    # new patient
    mc = parameterise_new_patient_object(
        engine,
        forecast_horizon,
        path_to_sql_files,
        new_patient_seed=new_patient_seed,
        patient_categoriser_seed=patient_categoriser_seed,
    )
    new_patient_function = mc.generate_new_patients

    # initial waiting list
    initial_waiting_list = get_initial_waiting_list(engine, path_to_sql_files)

    mridept = MRIDepartment(
        f"{path_to_sql_files}/transformed_mri_scanners.json",
        fu_rate,
        fu_rng,
        clinic_utilisation,
    )

    # resource matching
    resource_matching_function = mridept.match_mri_resource

    # priority order
    priority_order = [
        "Breach",
        "Max wait time",
        "Breach days",
        "Days waited",
        "Over minimum wait time",
        "Under maximum wait time",
    ]

    sim = parameterise_simulation(
        initial_waiting_list,
        new_patient_function,
        resource_matching_function,
        priority_order,
        dna_rate,
        cancellation_rate,
        forecast_horizon,
        rott_dist_params=rott_dist_params,
        rott_seed=rott_seed,
        capacity_seed=capacity_seed,
        dna_seed=dna_seed,
        cancellation_seed=cancellation_seed,
        max_wait_time=42,
        metrics=MRIMetrics,
    )

    return seed, sim, mridept
