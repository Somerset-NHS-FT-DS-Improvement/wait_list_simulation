from .simulation_components import PatientGenerator, Capacity, generate_simulation_graph, get_appointment_duration
from .patient_management.priority import PriorityCalculator

import sfttoolbox

import numpy as np


def parameterise_simulation(initial_waitlist, new_patient_function, resource_matching_function, priority_order,
                            dna_rate, cancellation_rate, emergency_rate, fu_rate, length_of_simulation, seed=None):
    # TODO: take in seeds for DNA, cancellations
    # TODO: take in emergency rate and seeds
    # TODO: take in FU rate and seeds

    seed = seed if seed else np.random.default_rng().integers(0, 2**32)
    # TODO: Update below with number of seeds..
    seeds = np.random.default_rng(seed).integers(0, 2**32, 10)

    # TODO: Update the seeds below
    # emergency_random_number_generator =

    pg = PatientGenerator(new_patient_function, initial_waitlist)

    pc = PriorityCalculator(priority_order)
    cap = Capacity(resource_matching_function, pc)

    G = generate_simulation_graph(cap, get_appointment_duration)

    return sfttoolbox.DES.Simulation(G, pg, 0), seed
