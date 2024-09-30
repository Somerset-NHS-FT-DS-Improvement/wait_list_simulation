import pandas as pd

import networkx as nx

import numpy as np


class PatientGenerator:
    def __init__(
        self, generate_new_patients, emergency_random_number_gen=None, start_id=0
    ):
        self.id = start_id
        self.generate_new_patients = generate_new_patients

        self.emergency_random_number_gen = (
            emergency_random_number_gen
            if emergency_random_number_gen
            else np.random.normal
        )

        # TODO: take in Emergency rate and seeds.

    def generate_patients(self, day_num, day):
        # Emergency and new patients dealt with through the interface with generate_new_patients
        patients = self.generate_new_patients(day_num)

        patients.loc[:, "Pathway"] = [[] for _ in range(patients.shape[0])]

        # TODO: fill in appointment_duration if not present, fill nans etc...

        patients.loc[:, "id"] = [*range(self.id, self.id + len(patients))]
        self.id += len(patients) - 1

        # Generator object
        iterable_object = patients.iterrows()
        return self.__yield_patient(iterable_object)

    def __yield_patient(self, iterable_object):
        for _, patient in iterable_object:
            yield patient


def get_appointment_duration(patient):
    return patient.duration_mins


class Capacity:
    def __init__(
        self,
        match_resource,
        prioritisation_calculator,
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
        initial_wait_list,
    ):
        self.wait_list = initial_wait_list
        self.prioritisation_calculator = prioritisation_calculator
        self.match_resource = match_resource

        self.dna_rate = dna_rate
        self.cancellation_rate = cancellation_rate
        self.emergency_rate = emergency_rate
        self.fu_rate = fu_rate
        self.rott_rate = rott_rate

        self.dna_rng = dna_rng
        self.cancellation_rng = cancellation_rng
        self.emergency_rng = emergency_rng
        self.fu_rng = fu_rng
        self.rott_rng = rott_rng

        self.wait_list["id"] = [*range(0, self.wait_list.shape[0])]
        self.wait_list["Pathway"] = [[] for _ in range(self.wait_list.shape[0])]

        self.metrics = {"maximum_wait_time": [], "wait_list_length": []}

    def get(self, resource, patient, day_num, day):
        self.wait_list = pd.concat([self.wait_list, patient.to_frame().T])
        self.wait_list.reset_index(drop=True)

        return True

    def update_day(self, day_num, day):
        # prioritise wait list
        indices = self.prioritisation_calculator.calculate_sorted_indices(
            self.wait_list
        )
        self.wait_list = self.wait_list.iloc[indices]

        # TODO: deal with fu, emergency, ROTT, DNA, cancellations, then update below
        patients_DNA_or_cancel = pd.DataFrame()

        patients_to_move_on = self.match_resource(self.wait_list)
        # TODO: add patients to have fu to the wait list with new FU label

        # Or take patients to move on, and randomly select to have FU appts?
        # Use a geometric distribution, 1/1-p = total num of desired appts, solve for p, then random number lower than p means a fu is done

        self.__update_metrics()
        self.wait_list["days waited"] += 1

        return patients_to_move_on

    def __update_metrics(self):
        self.metrics["maximum_wait_time"].append(self.wait_list["days waited"].max())
        self.metrics["wait_list_length"].append(self.wait_list.shape[0])


def generate_simulation_graph(capacity, resource):
    G = nx.DiGraph()
    G.add_node("Simulate", capacity=capacity, resource=resource)
    return G
