import numpy as np

from .patient_management.priority import PriorityCalculator
from .simulation_components import (Capacity, PatientGenerator,
                                    generate_simulation_graph,
                                    get_appointment_duration)


def parameterise_simulation(
    initial_waitlist,
    new_patient_function,
    resource_matching_function,
    priority_order,
    dna_rate,
    cancellation_rate,
    emergency_rate,
    fu_rate,
    rott_rate,
    length_of_simulation,
    dna_rng,
    cancellation_rng,
    emergency_rng,
    fu_rng,
    rott_rng,
):
    # TODO: Fix this hack... make sfttoolbox available outside of org?
    import sfttoolbox

    pg = PatientGenerator(new_patient_function, start_id=initial_waitlist.shape[0])

    pc = PriorityCalculator(priority_order)
    # TODO: Pass all the rates and rngs through
    cap = Capacity(
        resource_matching_function,
        pc,
        dna_rate,
        cancellation_rate,
        emergency_rate,
        fu_rate,
        rott_rate,
        dna_rng,
        cancellation_rng,
        emergency_rng,
        fu_rng,
        rott_rng,
        initial_waitlist,
    )

    G = generate_simulation_graph(cap, get_appointment_duration)

    return sfttoolbox.DES.Simulation(G, pg, length_of_simulation)


if __name__ == "__main__":
    # parameterise the mri new patient class -- Mon -- DONE
    # pass the new object to the above function -- Mon -- DONE

    # get the initial wait list together -- Mon -- DONE
    # pass to the above function -- Mon -- DONE

    # create a dummy resource matching -- Mon -- DONE

    # priority order -- Mon -- DONE
    # dna_rate, cancellation_rate, emergency_rate, fu_rate all set to 0 for now -- Mon -- DONE
    # length_of_simulation -- Mon -- DONE

    # run simulation always increasing waitlist -- Mon

    # Look at altering all the rates -- Mon

    # create metrics to track how much increased -- Tues

    # start to interface with webapp -- Tues

    pass
