from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .. import parameterise_simulation
from ..patient_management import Data, NewPatients
from ..patient_management.priority import PriorityCalculator
from ..patient_management.rott import RemovalOtherThanTreatment
from .department import MRIDepartment
from .metrics import MRIMetrics


def process_data(data: Data, max_wait_time: int) -> None:
    """
    Process the data by cleaning and filling missing values, and calculating wait times.

    Args:
        data (Data): The data object containing historic and current waiting lists.
        max_wait_time (int): The maximum wait time to be used in the priority calculation.
    """
    data.historic_waiting_list["priority"] = data.historic_waiting_list[
        "priority"
    ].str.strip()

    data.historic_waiting_list["priority"] = data.historic_waiting_list[
        "priority"
    ].fillna("Urgent")

    data.historic_waiting_list["duration_mins"] = data.historic_waiting_list[
        "duration_mins"
    ].fillna(30)

    # Doing this here means it's not required each time new patients are selected
    pc = PriorityCalculator([], max_wait_time)
    data.historic_waiting_list.loc[:, ["min_wait", "max_wait"]] = (
        pc.calculate_min_and_max_wait_times(data.historic_waiting_list)
    )

    data.current_waiting_list["priority"] = data.current_waiting_list[
        "priority"
    ].str.strip()

    # These values taken from a meeting with the MRI dept
    data.current_waiting_list["priority"] = data.current_waiting_list[
        "priority"
    ].fillna("Urgent")
    data.current_waiting_list["duration_mins"] = data.current_waiting_list[
        "duration_mins"
    ].fillna(30)


class MriSimulation:
    def __init__(
        self,
        path_to_sql_queries: str,
        dna_rate: Optional[float] = None,
        cancellation_rate: Optional[float] = None,
        discharge_rate: Optional[float] = None,
        fu_rate: Optional[float] = None,
        rott_params: Optional[Dict[str, float]] = None,
        clinic_utilisation: float = 1,
    ) -> Tuple[int, "Simulation"]:
        """
        Initialise the MriSimulation class.

        Args:
            path_to_sql_queries (str): Path to the SQL queries.
            dna_rate (Optional[float]): DNA rate. Default is None.
            cancellation_rate (Optional[float]): Cancellation rate. Default is None.
            discharge_rate (Optional[float]): Discharge rate of DNA/cancellation patients. Default is None.
            fu_rate (Optional[float]): Follow-up rate. Default is None.
            rott_params (Optional[Dict[str, float]]): mean and std_dev for rott generation. Default is None.
            clinic_utilisation (float): Clinic utilisation rate. Default is 1.

        Returns:
            Tuple[int, "Simulation"]: A tuple containing the seed and the simulation object.
        """
        self.forecast_horizon = 365
        self.max_wait_time = 42
        self.clinic_utilisation = clinic_utilisation
        self.path_to_sql_queries = path_to_sql_queries

        self.priority_order = [
            "MRI breaches",
            "MRI days until due",
            "Breach",
            "Max wait time",
            "Breach days",
            "Days waited",
            "Over minimum wait time",
            "Under maximum wait time",
        ]

        fu_file_name = None if fu_rate is not None else "MRI_fu_rate.sql"
        dna_file_name = None if dna_rate is not None else "MRI_dna_rate.sql"
        cancellation_file_name = (
            None if cancellation_rate is not None else "MRI_cancellation_rate.sql"
        )
        discharge_file_name = None if discharge_rate is not None else "MRI_discharge_rate.sql"
        rott_file_name = None if rott_params is not None else "MRI_rott_rate.sql"

        self.mri_data = Data(
            path_to_sql_queries=path_to_sql_queries,
            historic_waiting_list_file_name="MRI_historic_waiting_list.sql",
            num_new_refs_file_name="num_new_refs.sql",
            current_waiting_list_file_name="MRI_current_waiting_list.sql",
            dna_file_name=dna_file_name,
            cancellation_file_name=cancellation_file_name,
            discharge_file_name=discharge_file_name,
            fu_file_name=fu_file_name,
            rott_file_name=rott_file_name,
        )

        process_data(self.mri_data, self.max_wait_time)

        self.fu_rate = self.mri_data.fu_rate if fu_rate is None else fu_rate
        self.dna_rate = self.mri_data.dna_rate if dna_rate is None else dna_rate
        self.cancellation_rate = (
            self.mri_data.cancellation_rate
            if cancellation_rate is None
            else cancellation_rate
        )
        self.discharge_rate = self.mri_data.discharge_rate if discharge_rate is None else discharge_rate

        self.rott = RemovalOtherThanTreatment(horizon=self.forecast_horizon)
        if rott_params is None:
            self.rott.setup_distribution_from_data(**self.mri_data.rott)
        else:
            self.rott.setup_stochastic_distribution(**rott_params)

    def parameterise_simulation(
        self, seed: Optional[int] = None
    ) -> Tuple[int, "Simulation", MRIDepartment]:
        """
        Parameterise the simulation with the given seed.

        Args:
            seed (Optional[int]): Seed for the random number generator. Default is None.

        Returns:
            Tuple[int, "Simulation", Any]: A tuple containing the seed, the simulation object, and the MRI department object.
        """
        seed_gen = np.random.SeedSequence(seed)
        seed = seed_gen.entropy

        # seeds
        seeds = seed_gen.generate_state(8)

        new_patient_seed = (seeds[0],)
        patient_categoriser_seed = (seeds[1],)
        dna_seed = seeds[2]
        cancellation_seed = seeds[3]
        emergency_rng = np.random.default_rng(seed=seeds[4])
        fu_rng = np.random.default_rng(seed=seeds[5])
        rott_seed = seeds[6]
        capacity_seed = seeds[7]

        # new patient
        mc = NewPatients(
            self.mri_data.num_new_refs,
            self.mri_data.historic_waiting_list,
            self.forecast_horizon,
            new_patients_seed=new_patient_seed,
            patient_categoriser_seed=patient_categoriser_seed,
        )
        new_patient_function = mc.generate_new_patients

        mridept = MRIDepartment(
            f"{self.path_to_sql_queries}/transformed_mri_scanners.json",
            self.fu_rate,
            fu_rng,
            self.clinic_utilisation,
        )

        self.rott.seed = rott_seed

        # resource matching
        resource_matching_function = mridept.match_mri_resource

        sim = parameterise_simulation(
            self.mri_data.current_waiting_list,
            new_patient_function,
            resource_matching_function,
            self.priority_order,
            self.dna_rate,
            self.cancellation_rate,
            self.discharge_rate,
            self.forecast_horizon,
            rott_object=self.rott,
            capacity_seed=capacity_seed,
            dna_seed=dna_seed,
            cancellation_seed=cancellation_seed,
            max_wait_time=self.max_wait_time,
            metrics=MRIMetrics,
            exta_priorities=self._extra_priorities,
        )

        return seed, sim, mridept

    def _extra_priorities(self, df: pd.DataFrame) -> dict[str, List[int]]:
        """
        Calculate extra priorities for the MRI simulation.

        Args:
            df (pd.DataFrame): Dataframe containing patient data.

        Returns:
            dict[str, Any]: A dictionary containing extra priorities.
        """
        return {
            "MRI days until due": -(df["days waited"] - df["days_until_due"]).fillna(0),
            "MRI breaches": ~(
                (df["days waited"] - df["days_until_due"].fillna(0)) > 42
            ),
        }
