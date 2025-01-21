from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd

from .. import parameterise_simulation
from ..patient_management import Data, NewPatients
from ..patient_management.rott import RemovalOtherThanTreatment
from .resource_match import OutpatientResourceMatcher


# def parameterise_new_patient_object(
#     engine: sa.engine.Engine,
#     forecast_horizon: int,
#     treatment_function_code: int,
#     new_patient_seed: int = None,
#     patient_categoriser_seed: int = None,
#     site: str = "mph",
# ) -> "NewOutpatients":
#     """
#     Creates and returns an instance of the NewOutpatients object.
#
#     Args:
#         engine (sa.engine.Engine): Database engine to connect to the SQL database.
#         forecast_horizon (int): The number of periods (e.g., days, weeks) over which the forecast is made.
#
#         new_patient_seed (int, optional): Seed for random generation of new patients. Defaults to None.
#         patient_categoriser_seed (int, optional): Seed for categorizing new patients. Defaults to None.
#
#     Returns:
#         NewOutpatients: An instance of NewOutpatients class initialized with required data.
#     """
#
#     mc = NewPatients(
#         pd.read_sql(
#             f"Execute [wl].[opa_admissions] @site={site}, @tfc='{treatment_function_code}'",
#             engine,
#         ),
#         pd.read_sql(
#             f"Execute [wl].[current_opa_waiting_list] @site={site}, @tfc='{treatment_function_code}'",
#             engine,
#         ),
#         # TODO: Update this!
#         # pd.read_sql(
#         #     open(f"{path_to_sql_files}/MRI_historic_waiting_list.sql", "r").read(),
#         #     engine,
#         # ),
#         forecast_horizon,
#         new_patients_seed=new_patient_seed,
#         patient_categoriser_seed=patient_categoriser_seed,
#     )
#     return mc


# def get_initial_waiting_list(
#     engine: sa.engine.Engine, treatment_function_code: int, site: str = None
# ) -> pd.DataFrame:
#     """
#     Retrieves the initial OP waiting list from the database.
#
#     Args:
#         engine (sa.engine.Engine): Database engine to connect to the SQL database.
#
#     Returns:
#         pd.DataFrame: A DataFrame containing the initial OP waiting list.
#     """
#     return pd.read_sql(
#         f"Execute [wl].[current_opa_waiting_list] @site={site}, @tfc='{treatment_function_code}'",
#         engine,
#     )


# def resource_dummy_function(in_list):
#     num_patients_to_remove = np.random.randint(0, 10)
#     return in_list.iloc[:num_patients_to_remove].index


# def setup_op_simulation(
#     path_to_sql_files: str, treatment_function_code: int, seed: int = None
# ) -> tuple[int, "Simulation"]:
#     """
#     Sets up and initializes the OP simulation.
#
#     Args:
#         path_to_sql_files (str): Path to the SQL files and other necessary setup files.
#
#     Returns:
#         tuple[int, Simulation]: A tuple containing the seed used for the simulation and the initialized simulation object.
#     """
#     engine = sa.create_engine(open(f"{path_to_sql_files}/engine.txt", "r").read())
#
#     seed = np.random.default_rng().integers(0, 2**32) if not seed else seed
#
#     # seeds
#     seeds = np.random.default_rng(seed).integers(0, 2**32, 8)
#
#     # length_of_simulation
#     forecast_horizon = 5
#
#     # new patient
#     mc = parameterise_new_patient_object(
#         engine,
#         forecast_horizon,
#         treatment_function_code,
#         new_patient_seed=seeds[0],
#         patient_categoriser_seed=seeds[1],
#     )
#     new_patient_function = mc.generate_new_patients
#
#     # initial waiting list
#     initial_waiting_list = get_initial_waiting_list(engine, treatment_function_code)
#
#     # resource matching
#     oprm = OutpatientResourceMatcher(engine, 110)
#     resource_matching_function = oprm.match_resource
#
#     # priority order
#     priority_order = [
#         "A&E patients",
#         "inpatients",
#         "Breach",
#         "Days waited",
#         "Over minimum wait time",
#         "Under maximum wait time",
#     ]
#
#     # dna_rate, cancellation_rate, emergency_rate, fu_rate
#     dna_rate = 0
#     cancellation_rate = 0
#     emergency_rate = 0
#     fu_rate = 0
#     dna_rng = np.random.default_rng(seed=seeds[2])
#     cancellation_rng = np.random.default_rng(seed=seeds[3])
#     emergency_rng = np.random.default_rng(seed=seeds[4])
#     fu_rng = np.random.default_rng(seed=seeds[5])
#     rott_dist_params = {"mean": 1, "stddev": 1}
#
#     sim = parameterise_simulation(
#         initial_waiting_list,
#         new_patient_function,
#         resource_matching_function,
#         priority_order,
#         dna_rate,
#         cancellation_rate,
#         forecast_horizon,
#         rott_dist_params=rott_dist_params,
#         rott_seed=seeds[6],
#         capacity_seed=seeds[7],
#     )
#
#     return seed, sim




class OPSimulation:
    def __init__(
            self,
            path_to_sql_queries: str,
            site: str,
            tfc: int,
            dna_rate: Optional[float] = None,
            cancellation_rate: Optional[float] = None,
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
            fu_rate (Optional[float]): Follow-up rate. Default is None.
            rott_params (Optional[Dict[str, float]]): mean and std_dev for rott generation. Default is None.
            clinic_utilisation (float): Clinic utilisation rate. Default is 1.

        Returns:
            Tuple[int, "Simulation"]: A tuple containing the seed and the simulation object.
        """
        self.forecast_horizon = 365
        self.clinic_utilisation = clinic_utilisation
        self.path_to_sql_queries = path_to_sql_queries

        self.priority_order = [
            "A&E patients",
            "inpatients",
            "Breach",
            "Days waited",
            "Over minimum wait time",
            "Under maximum wait time",
        ]

        fu_file_name = None if fu_rate is not None else "op_fu_rate.sql"
        dna_file_name = None if dna_rate is not None else "op_dna_rate.sql"
        cancellation_file_name = (
            None if cancellation_rate is not None else "op_cancellation_rate.sql"
        )
        rott_file_name = None if rott_params is not None else "op_rott_params.sql"

        self.op_data = OPData(
            path_to_sql_queries=path_to_sql_queries,
            site=site,
            tfc=tfc,
            historic_waiting_list_file_name="op_historic_waiting_list.sql",
            num_new_refs_file_name="num_new_refs.sql",
            current_waiting_list_file_name="op_current_waiting_list.sql",
            dna_file_name=dna_file_name,
            cancellation_file_name=cancellation_file_name,
            fu_file_name=fu_file_name,
            rott_file_name=rott_file_name,
        )

        self.fu_rate = self.op_data.fu_rate if fu_rate is None else fu_rate
        self.dna_rate = self.op_data.dna_rate if dna_rate is None else dna_rate
        self.cancellation_rate = (
            self.op_data.cancellation_rate
            if cancellation_rate is None
            else cancellation_rate
        )

        self.rott = RemovalOtherThanTreatment(horizon=self.forecast_horizon)
        if rott_params is None:
            # self.rott.setup_distribution_from_data(**self.op_data.rott)
            self.rott.setup_stochastic_distribution(**self.op_data.rott)
        else:
            self.rott.setup_stochastic_distribution(**rott_params)

    def parameterise_simulation(
            self, seed: Optional[int] = None
    ) -> Tuple[int, "Simulation", OutpatientResourceMatcher]:
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

        new_patient_seed = seeds[0]
        patient_categoriser_seed = seeds[1]
        dna_seed = seeds[2]
        cancellation_seed = seeds[3]
        emergency_rng = np.random.default_rng(seed=seeds[4])
        fu_rng = np.random.default_rng(seed=seeds[5])
        rott_seed = seeds[6]
        capacity_seed = seeds[7]

        mc = NewPatients(
            self.op_data.num_new_refs,
            self.op_data.historic_waiting_list,
            self.forecast_horizon,
            new_patients_seed=new_patient_seed,
            patient_categoriser_seed=patient_categoriser_seed,
        )
        new_patient_function = mc.generate_new_patients

        opdept = OutpatientResourceMatcher(
            self.op_data.op_clinic_slots,
            self.fu_rate,
            fu_rng
        )

        self.rott.seed = rott_seed

        resource_matching_function = opdept.match_resource

        sim = parameterise_simulation(
            self.op_data.current_waiting_list,
            new_patient_function,
            resource_matching_function,
            self.priority_order,
            self.dna_rate,
            self.cancellation_rate,
            self.forecast_horizon,
            rott_object=self.rott,
            capacity_seed=capacity_seed,
            dna_seed=dna_seed,
            cancellation_seed=cancellation_seed,
        )

        return seed, sim, opdept

class OPData(Data):
    def __init__(
        self,
        site,
        tfc,
        path_to_sql_queries,
        historic_waiting_list_file_name,
        num_new_refs_file_name,
        current_waiting_list_file_name,
        fu_file_name=None,
        dna_file_name=None,
        cancellation_file_name=None,
        rott_file_name=None
    ):
        self.site = site
        self.tfc = tfc

        super().__init__(
            path_to_sql_queries,
            historic_waiting_list_file_name,
            num_new_refs_file_name,
            current_waiting_list_file_name,
            fu_file_name,
            dna_file_name,
            cancellation_file_name,
            rott_file_name
        )

        # TODO: This is very slow!
        self.op_clinic_slots = self._run_sql("op_clinic_slots.sql")

    def _run_sql(self, filename):
        return pd.read_sql(
            open(f"{self.path_to_sql_queries}/{filename}", "r").read().format(site=self.site, tfc=self.tfc),
            self.engine,
        )

    # TODO: add in a pickler?
