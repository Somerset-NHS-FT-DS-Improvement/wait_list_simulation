from datetime import datetime
from typing import Callable, List, Optional

import pandas as pd
import sfttoolbox

from .patient_management.priority import PriorityCalculator
from .patient_management.rott import RemovalOtherThanTreatment
from .simulation_components import (Capacity, Metrics, PatientGenerator,
                                    generate_simulation_graph,
                                    get_appointment_duration)

__all__ = ["parameterise_simulation"]


def parameterise_simulation(
    initial_waitlist: pd.DataFrame,
    new_patient_function: Callable,
    resource_matching_function: Callable,
    priority_order: List,
    dna_rate: float,
    cancellation_rate: float,
    discharge_rate: float,
    length_of_simulation: int,
    rott_object: RemovalOtherThanTreatment,
    capacity_seed: Optional[int] = None,
    dna_seed: Optional[int] = None,
    cancellation_seed: Optional[int] = None,
    max_wait_time: int = None,
    metrics=Metrics,
    exta_priorities=None,
) -> sfttoolbox.DES.Simulation:
    """
    Parameterizes the simulation by setting up the patient generator, priority calculator,
    and capacity model, as well as configuring removal due to reasons other than treatment (ROTT).

    Args:
        initial_waitlist (pd.DataFrame): DataFrame representing the initial list of patients waiting for treatment.
        new_patient_function (Callable): Function that generates new patients.
        resource_matching_function (Callable): Function used to match resources with patient requirements.
        priority_order (Any): Defines the priority ordering for the patients.
        dna_rate (float): The "Did Not Attend" (DNA) rate for missed appointments.
        cancellation_rate (float): The rate of cancellations by patients.
        discharge_rate (float): The rate at which patients discharge after DNA/cancellation.
        length_of_simulation (int): The total number of simulation days.
        rott_object (RemovalOtherThanTreatment): A parameterised rott object.
        capacity_seed (Optional[int]): Seed for random number generation in capacity calculations.
        dna_seed (Optional[int]): Seed for random number generation in DNA calculations.
        cancellation_seed (Optional[int]): Seed for random number generation in cancellation calculations.

    Returns:
        Simulation: Returns the initialized custom simulation object.

    Raises:
        Exception: If neither `rott_sql_query` nor `rott_dist_params` are provided.
    """
    pg = PatientGenerator(new_patient_function, start_id=initial_waitlist.shape[0])

    pc = PriorityCalculator(priority_order, max_wait_time, exta_priorities)
    initial_waitlist.loc[:, ["min_wait", "max_wait"]] = (
        pc.calculate_min_and_max_wait_times(initial_waitlist)
    )

    rott_removals = rott_object.return_number_of_removals()

    cap = Capacity(
        resource_matching_function,
        pc,
        dna_rate,
        cancellation_rate,
        discharge_rate,
        initial_waitlist,
        rott_removals,
        seed=capacity_seed,
        dna_seed=dna_seed,
        cancellation_seed=cancellation_seed,
        metrics=metrics,
    )

    G = generate_simulation_graph(cap, get_appointment_duration)

    sim = sfttoolbox.DES.Simulation(
        G, pg, length_of_simulation, start_day=datetime.now().strftime("%a")
    )

    return sim
