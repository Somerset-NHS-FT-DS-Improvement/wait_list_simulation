from datetime import datetime
from typing import Callable, Dict, List, Optional

import pandas as pd
import sfttoolbox
from sqlalchemy.engine import Engine

from .patient_management.priority import PriorityCalculator
from .patient_management.rott import RemovalOtherThanTreatment
from .simulation_components import (Capacity, PatientGenerator,
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
    length_of_simulation: int,
    rott_sql_query: Optional[Dict[str, Engine]] = None,
    rott_dist_params: Optional[Dict[str, float]] = None,
    rott_seed: Optional[int] = None,
    capacity_seed: Optional[int] = None,
    dna_seed: Optional[int] = None,
    cancellation_seed: Optional[int] = None,
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
        length_of_simulation (int): The total number of simulation days.
        rott_sql_query (Optional[Dict[str, Any]]): SQL query information for ROTT distribution, if provided.
        rott_dist_params (Optional[Dict[str, float]]): Parameters for a stochastic distribution for ROTT.
        rott_seed (Optional[int]): Seed for random number generation in ROTT.
        capacity_seed (Optional[int]): Seed for random number generation in capacity calculations.
        dna_seed (Optional[int]): Seed for random number generation in DNA calculations.
        cancellation_seed (Optional[int]): Seed for random number generation in cancellation calculations.

    Returns:
        Simulation: Returns the initialized custom simulation object.

    Raises:
        Exception: If neither `rott_sql_query` nor `rott_dist_params` are provided.
    """
    pg = PatientGenerator(new_patient_function, start_id=initial_waitlist.shape[0])

    pc = PriorityCalculator(priority_order)

    rott = RemovalOtherThanTreatment(length_of_simulation, seed=rott_seed)
    if rott_sql_query:
        rott.setup_sql_distribution(
            sql_engine=rott_sql_query["engine"], query_string=rott_sql_query["query"]
        )
    elif rott_dist_params:
        rott.setup_stochastic_distribution(
            mean=rott_dist_params["mean"], std_dev=rott_dist_params["stddev"]
        )
    else:
        raise Exception("Either rott_sql_query or rott_dist_params must be defined.")

    rott_removals = rott.return_number_of_removals()

    cap = Capacity(
        resource_matching_function,
        pc,
        dna_rate,
        cancellation_rate,
        initial_waitlist,
        rott_removals,
        seed=capacity_seed,
        dna_seed=dna_seed,
        cancellation_seed=cancellation_seed,
    )

    G = generate_simulation_graph(cap, get_appointment_duration)

    sim = sfttoolbox.DES.Simulation(
        G, pg, length_of_simulation, start_day=datetime.now().strftime("%a")
    )

    return sim
