from .patient_management.priority import PriorityCalculator
from .simulation_components import (Capacity, PatientGenerator,
                                    generate_simulation_graph,
                                    get_appointment_duration)
from .patient_management.rott import RemovalOtherThanTreatment

import sfttoolbox


__all__ = ["parameterise_simulation"]


def parameterise_simulation(
    initial_waitlist,
    new_patient_function,
    resource_matching_function,
    priority_order,
    dna_rate,
    cancellation_rate,
    length_of_simulation,
    rott_sql_query = None,
    rott_dist_params = None,
    rott_seed = None,
    capacity_seed = None,
    dna_seed = None,
    cancellation_seed = None
):
    pg = PatientGenerator(new_patient_function, start_id=initial_waitlist.shape[0])

    pc = PriorityCalculator(priority_order)

    rott = RemovalOtherThanTreatment(length_of_simulation, seed=rott_seed)
    if rott_sql_query:
        rott.setup_sql_distribution(sql_engine=rott_sql_query["engine"], query_string=rott_sql_query["query"])
    elif rott_dist_params:
        rott.setup_stochastic_distribution(mean=rott_dist_params["mean"], std_dev=rott_dist_params["stddev"])
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
        seed = capacity_seed,
        dna_seed = dna_seed,
        cancellation_seed = cancellation_seed
    )

    G = generate_simulation_graph(cap, get_appointment_duration)

    return sfttoolbox.DES.Simulation(G, pg, length_of_simulation)
