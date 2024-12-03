from collections import defaultdict
from typing import Any, Callable, Generator

import networkx as nx
import numpy as np
import pandas as pd


class PatientGenerator:
    """
    A class to generate new patients for each simulation day and assign them unique IDs.

    Args:
        generate_new_patients (Callable[[int], pd.DataFrame]): A function that generates new patients,
            returning a DataFrame based on the day number.
        start_id (int, optional): The starting ID for patients. Defaults to 0.

    Attributes:
        id (int): The current patient ID, incremented for each new patient.
        generate_new_patients (Callable[[int], pd.DataFrame]): A function that generates new patients.
    """

    def __init__(
        self, generate_new_patients: Callable[[int], pd.DataFrame], start_id: int = 0
    ) -> None:
        self.id = start_id
        self.generate_new_patients = generate_new_patients

    def generate_patients(
        self, day_num: int, day: str
    ) -> Generator[pd.Series, None, None]:
        """
        Generates patients for a given day by calling the provided new patient generation function.
        It assigns a unique ID to each patient and prepares the patient data.

        Args:
            day_num (int): The simulation day number used to generate new patients.
            day (str): Represents the day in the simulation (not directly used but passed as part of the simulation loop).

        Returns:
            Generator[pd.Series, None, None]: A generator that yields individual patient rows.
        """
        # Emergency and new patients dealt with through the interface with generate_new_patients
        patients = self.generate_new_patients(day_num)

        patients.loc[:, "pathway"] = [[] for _ in range(patients.shape[0])]

        patients.loc[:, "id"] = [*range(self.id, self.id + len(patients))]
        self.id += len(patients) - 1

        # Generator object
        iterable_object = patients.iterrows()
        return self.__yield_patient(iterable_object)

    def __yield_patient(
        self, iterable_object: pd.core.indexes.base.IndexOpsMixin
    ) -> Generator[pd.Series, None, None]:
        """
        Private method that yields each patient from the iterable object.

        Args:
            iterable_object (pd.core.indexes.base.IndexOpsMixin): An iterable object that contains rows of patient data.

        Yields:
            pd.Series: A row of patient data representing an individual patient.
        """
        for _, patient in iterable_object:
            yield patient


def get_appointment_duration(patient: pd.Series) -> int:
    """
    Retrieves the appointment duration for a given patient.

    Args:
        patient (pd.Series): A pandas Series representing a patient's data, which must contain
                             the 'duration_mins' attribute.

    Returns:
        int: The duration of the appointment in minutes.
    """
    return patient.duration_mins


class Metrics:
    def __init__(self):
        self.metrics = defaultdict(lambda: [])

    def update_metrics(
        self,
        capacity_object,
        num_patients_seen: int,
        num_dnas: int,
        num_cancellations: int,
    ) -> None:
        """
        Updates the metrics for the current day.

        Args:
            num_patients_seen (int): The number of patients seen on that day.
            num_dnas (int): The number of patients who did not attend (DNA).
            num_cancellations (int): The number of cancellations for that day.
        """
        self.metrics["maximum_wait_time"].append(
            capacity_object.wait_list["days waited"].max()
        )
        self.metrics["wait_list_length"].append(capacity_object.wait_list.shape[0])
        self.metrics["num_patients_seen"].append(num_patients_seen)
        self.metrics["num_dnas"].append(num_dnas)
        self.metrics["num_cancellations"].append(num_cancellations)
        # self.metrics["num_breaches"].append(capacity_object.__calculate_breaches())
        self.metrics["median_wait_times_by_priority"].append(
            capacity_object.wait_list.groupby("priority")["days waited"]
            .median()
            .to_dict()
        )
        self.metrics["max_wait_times_by_priority"].append(
            capacity_object.wait_list.groupby("priority")["days waited"].max().to_dict()
        )


class Capacity:
    def __init__(
        self,
        match_resource,
        prioritisation_calculator,
        dna_rate,
        cancellation_rate,
        initial_wait_list,
        rott_removals,
        seed=None,
        dna_seed=None,
        cancellation_seed=None,
        metrics=Metrics,
    ):
        """
        A class to manage patient capacity and scheduling within a healthcare simulation.

        Args:
            match_resource (Callable[[pd.DataFrame, Any, int], Any]): A function that matches resources to patients,
                given a DataFrame of patients, the current day, and the day number.
            prioritisation_calculator (Any): An object responsible for calculating patient prioritization.
            dna_rate (float): The rate of patients who do not attend (DNA) their appointments.
            cancellation_rate (float): The rate of patients who cancel their appointments.
            initial_wait_list (pd.DataFrame): A DataFrame representing the initial list of patients waiting for treatment.
            rott_removals (list[int]): A list of the number of removals due to reasons other than treatment (ROTT)
                for each day in the simulation.
            seed (Optional[int]): Random seed for the RNG (default is None).
            dna_seed (Optional[int]): Random seed for the DNA RNG (default is None).
            cancellation_seed (Optional[int]): Random seed for the cancellation RNG (default is None).

        Attributes:
            wait_list (pd.DataFrame): DataFrame representing the current wait list of patients.
            prioritisation_calculator (Any): The prioritisation calculator object.
            match_resource (Callable): Function to match resources to patients.
            dna_rate (float): Rate of patients not attending their appointments.
            cancellation_rate (float): Rate of patients cancelling their appointments.
            rott_removals (list[int]): Removals due to reasons other than treatment for each day.
            rng (np.random.Generator): RNG for general use.
            dna_rng (np.random.Generator): RNG for DNA calculations.
            cancellation_rng (np.random.Generator): RNG for cancellation calculations.
            metrics (Metrics): Metrics class, used to track metrics through the simulation.
        """
        self.wait_list_holder = []
        self.wait_list = initial_wait_list
        self.prioritisation_calculator = prioritisation_calculator
        self.match_resource = match_resource

        self.dna_rate = dna_rate
        self.cancellation_rate = cancellation_rate

        self.rott_removals = rott_removals

        self.rng = np.random.default_rng(seed)
        self.dna_rng = np.random.default_rng(dna_seed)
        self.cancellation_rng = np.random.default_rng(cancellation_seed)

        self.wait_list["id"] = [*range(0, self.wait_list.shape[0])]
        self.wait_list["pathway"] = [[] for _ in range(self.wait_list.shape[0])]

        # Initialise the object
        self.metrics = metrics()

    def get(self, resource: Any, patient: pd.Series, day_num: int, day: Any) -> bool:
        """
        Adds a patient to the wait list and resets the index.

        Args:
            resource (Any): The resource associated with the patient.
            patient (pd.Series): A Series representing the patient data to be added.
            day_num (int): The current day number of the simulation.
            day (Any): The current day in the simulation (not directly used).

        Returns:
            bool: Always returns True upon successfully adding the patient.
        """
        self.wait_list_holder.append(patient)

        return True

    def update_day(self, day_num: int, day: str) -> Generator[pd.Series, None, None]:
        """
        Updates the state for the current day, prioritizing the wait list, matching resources,
        handling non-attendance and cancellations, and updating metrics.

        Args:
            day_num (int): The current day number in the simulation.
            day (Any): Represents the day in the simulation (not directly used but passed).

        Returns:
            Generator[pd.Series, None, None]: A generator that yields patients who are seen on that day.
        """
        self.wait_list = pd.concat([self.wait_list, pd.DataFrame(self.wait_list_holder)]).reset_index(drop=True)
        self.wait_list_holder = []
        # prioritise wait list
        indices = self.prioritisation_calculator.calculate_sorted_indices(
            self.wait_list
        )
        self.wait_list = self.wait_list.iloc[indices]

        patients_to_move_on_indices, fu_patients = self.match_resource(
            self.wait_list, day, day_num
        )
        num_patients_seen = len(patients_to_move_on_indices)

        num_dnas = 0
        if self.dna_rate:
            patients_to_move_on_indices, num_dnas = self.__calculate_non_attendance(
                patients_to_move_on_indices, self.dna_rate, self.dna_rng
            )

        num_cancellations = 0
        if self.cancellation_rate:
            (
                patients_to_move_on_indices,
                num_cancellations,
            ) = self.__calculate_non_attendance(
                patients_to_move_on_indices,
                self.cancellation_rate,
                self.cancellation_rng,
            )

        patients_to_move_on = self.wait_list[
            self.wait_list.index.isin(patients_to_move_on_indices)
        ]
        self.wait_list.drop(index=patients_to_move_on_indices, inplace=True)

        # All FU logic is bespoke, and added in the resource matching function
        if fu_patients is not None:
            self.wait_list = pd.concat([self.wait_list, fu_patients])

        rott_patients = self.rng.choice(
            self.wait_list.index, self.rott_removals[day_num]
        )
        self.wait_list.drop(index=rott_patients, inplace=True)

        self.metrics.update_metrics(
            self, num_patients_seen, num_dnas, num_cancellations
        )
        self.wait_list["days waited"] += 1

        iterable_patients = patients_to_move_on.iterrows()
        return self.__yield_patient(iterable_patients)

    def __calculate_non_attendance(
        self,
        patients_assigned_slots_indices: list[int],
        rate: float,
        rng: np.random.Generator,
    ) -> tuple[list[int], int]:
        """
        Calculates the number of patients who will not attend based on the given rate.

        Args:
            patients_assigned_slots_indices (list[int]): List of indices for patients assigned slots.
            rate (float): The rate of non-attendance (DNA).
            rng (np.random.Generator): Random number generator for selecting non-attending patients.

        Returns:
            tuple[list[int], int]: A tuple containing the updated list of patient indices
            and the number of patients who did not attend.
        """
        num_non_attend = int(len(patients_assigned_slots_indices) * (rate / 100))
        patients_not_attending_indices = rng.choice(
            patients_assigned_slots_indices, num_non_attend, replace=False
        )

        for patient_index in patients_not_attending_indices:
            patients_assigned_slots_indices.remove(patient_index)

        return patients_assigned_slots_indices, num_non_attend

    def __calculate_breaches(self) -> dict:
        """
        Calculates the number of breaches based on the maximum wait times for different priorities.

        Returns:
            dict: A dictionary with priority as keys and counts of breaches as values.
        """
        min_max_wait_times = (
            self.prioritisation_calculator.calculate_min_and_max_wait_times(
                self.wait_list
            )
        )

        return (
            self.wait_list[self.wait_list["days waited"] > min_max_wait_times[:, 1]]
            .groupby("priority")["id"]
            .count()
            .to_dict()
        )

    def __yield_patient(
        self, iterable_object: pd.core.indexes.base.IndexOpsMixin
    ) -> Generator[pd.Series, None, None]:
        """
        Yields patients from the iterable object.

        Args:
            iterable_object (pd.core.indexes.base.IndexOpsMixin): An iterable object containing patient data.

        Yields:
            pd.Series: A row of patient data representing an individual patient.
        """
        for _, patient in iterable_object:
            yield patient


def generate_simulation_graph(capacity: Any, resource: Any) -> nx.DiGraph:
    """
    Generates a directed graph for the simulation using NetworkX.

    Args:
        capacity (Any): The capacity object used for the simulation.
        resource (Any): The resource associated with the simulation.

    Returns:
        nx.DiGraph: A directed graph representing the simulation, with a node for simulation parameters.
    """
    G = nx.DiGraph()
    G.add_node("Simulate", capacity=capacity, resource=resource)
    return G
