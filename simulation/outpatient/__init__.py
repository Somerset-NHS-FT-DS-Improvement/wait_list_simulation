from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import datetime

from .. import parameterise_simulation
from ..patient_management import Data, NewPatients
from ..patient_management.priority import PriorityCalculator
from ..patient_management.rott import RemovalOtherThanTreatment
from .resource_match import OutpatientResourceMatcher


class OPSimulation:
    def __init__(
            self,
            path_to_sql_queries: str,
            site: str,
            tfc: int,
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
            discharge_rate (Optional[float]): The rate at which dna/cancelled patients discharge.
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
            "Breach percentage",
            "Days waited",
            "Over minimum wait time",
            "Under maximum wait time",
        ]

        fu_file_name = None if fu_rate is not None else "op_fu_rate.sql"
        dna_file_name = None if dna_rate is not None else "op_dna_rate.sql"
        cancellation_file_name = (
            None if cancellation_rate is not None else "op_cancellation_rate.sql"
        )
        discharge_file_name = None if discharge_rate else "op_discharge_rate.sql"
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
            discharge_file_name=discharge_file_name,
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
        self.discharge_rate = self.op_data.discharge_rage if discharge_rate is None else discharge_rate

        self.rott = RemovalOtherThanTreatment(horizon=self.forecast_horizon)
        if rott_params is None:
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

        mc = OPNewPatient(
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
            self.discharge_rate,
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
        discharge_file_name=None,
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
            discharge_file_name,
            rott_file_name
        )
        # TODO: This is very slow!
        # self.op_clinic_slots = self._run_sql("op_clinic_slots.sql")


        df = pd.read_csv(f"{self.path_to_sql_queries}/wl_stats.csv")
        self.num_new_refs = self._num_new_refs_hack(df)
        self.op_clinic_slots = self._op_clinic_slots(df)

        self.current_waiting_list["days waited"] = np.clip(self.current_waiting_list["days waited"].fillna(0), 0, np.inf)

        pc = PriorityCalculator([])
        self.historic_waiting_list.loc[:, ["min_wait", "max_wait"]] = (
            pc.calculate_min_and_max_wait_times(self.historic_waiting_list)
        )

        self.current_waiting_list["ApptBookDate"] = pd.to_datetime(self.current_waiting_list["ApptBookDate"])
        self.current_waiting_list["sim_day_appt_due"] = (datetime.datetime.now() - self.current_waiting_list["ApptBookDate"]).dt.days

        self.historic_waiting_list["census"] = pd.to_datetime(self.historic_waiting_list["census"])
        self.historic_waiting_list["ApptBookDate"] = pd.to_datetime(self.historic_waiting_list["ApptBookDate"])
        self.historic_waiting_list["sim_day_appt_due"] = (self.historic_waiting_list["census"] - self.historic_waiting_list["ApptBookDate"]).dt.days

    def _num_new_refs_hack(self, df):
        num_new_refs_df = df[(df["type"] == "additions") & (df["appointment_type"] == "first") & (df["treatment_function_code"] == self.tfc)]
        refs_divided, refs_remainder = divmod(num_new_refs_df["count"], 7)

        num_new_refs = num_new_refs_df[["census", "count"]].set_index("census")
        num_new_refs.index = pd.to_datetime(num_new_refs.index)

        num_new_refs.loc[:, "count"] = refs_divided
        num_new_refs = num_new_refs.asfreq('d').ffill()

        num_new_refs.loc[num_new_refs.index.dayofweek == 6, "count"] += refs_remainder.values
        return num_new_refs.reset_index().rename({"census": "dst", "count":"refs"}, axis=1)

    def _op_clinic_slots(self, df):
        clinic_slots = df[(df["type"] == "removals") & (df["treatment_function_code"] == self.tfc)].pivot_table(columns="appointment_type", values="count", index="census")

        clinic_slots.index = pd.to_datetime(clinic_slots.index)

        clinic_slots_divided, clinic_slots_remainder = divmod(clinic_slots, 7)
        clinic_slots_divided = clinic_slots_divided.asfreq('d').ffill()

        clinic_slots_divided.loc[clinic_slots_divided.index.dayofweek == 6, :] += clinic_slots_remainder.values


        clinic_slots_divided = clinic_slots_divided.reset_index().rename({"census":"SessionDate"}, axis=1)
        clinic_slots_divided["unknown"] = 0
        clinic_slots_divided["Total_slots"] = clinic_slots_divided[["first", "followup"]].sum(axis=1)

        return clinic_slots_divided


    def _run_sql(self, filename):
            return pd.read_sql(
                open(f"{self.path_to_sql_queries}/{filename}", "r").read().format(site=self.site, tfc=self.tfc),
                self.engine,
            )

    # TODO: add in a pickler?


class OPNewPatient(NewPatients):
    def __init__(self,
                 historic_num_refs: pd.Series,
                 historic_data: pd.DataFrame,
                 forecast_horizon: int,
                 new_patients_seed: Optional[int] = None,
                 patient_categoriser_seed: Optional[int] = None):
        super().__init__(historic_num_refs, historic_data, forecast_horizon, new_patients_seed, patient_categoriser_seed)

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
            df["sim_day_appt_due"] += day_number

        return df
