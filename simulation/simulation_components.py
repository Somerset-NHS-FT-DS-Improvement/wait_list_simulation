import pandas as pd

import networkx as nx

import numpy as np

class PatientGenerator:
    def __init__(self, generate_new_patients, initial_waitlist=pd.DataFrame(), emergency_random_number_gen=None):
        self.id = 0
        self.initial_waitlist = initial_waitlist
        self.generate_new_patients = generate_new_patients

        self.emergency_random_number_gen = emergency_random_number_gen if emergency_random_number_gen else np.random.normal

        # TODO: take in Emergency rate and seeds.

    def generate_patients(self, day_num, day):
        # Emergency and new patients dealt with through the interface with generate_new_patients
        if day_num != 0:
            patients = self.generate_new_patients()
        else:
            patients = self.initial_waitlist

        patients["Pathway"] = [[] for _ in range(patients.shape[0])]

        # TODO: fill in appointment_duration if not present, fill nans etc...

        patients["id"] = [*range(self.id, self.id + len(patients))]
        self.id += len(patients)-1

        # Generator object
        return patients.iterrows()


def get_appointment_duration(patient):
    return patient.duration_mins


class Capacity:
    def __init__(self, match_resource, prioritisation_calculator):
        self.wait_list = []
        self.prioritisation_calculator = prioritisation_calculator
        self.match_resource = match_resource

        # TODO: Need to take in seeds for DNA, cancellations

    def get(self, resource, patient, day_num, day):
        self.wait_list.append(patient) # should this be a dataframe to be compatible with the prioritisation?

        return True

    def update_day(self, day_num, day):
        # TODO: Take in Fu rate? Or prob

        # prioritise wait list
        self.prioritisation_calculator.calculate_sorted_indices(self.wait_list)
        self.wait_list = self.wait_list.iloc[self.prioritisation_calculator.sorted_indices]


        # TODO: deal with ROTT, DNA, cancellations, then update below
        patients_DNA_or_cancel = pd.DataFrame()

        patients_to_move_on, patients_to_have_fu = self.match_resource(self.wait_list, patients_DNA_or_cancel)
        # TODO: add patients to have fu to the wait list with new FU label

        # Or take patients to move on, and randomly select to have FU appts?
        # Use a geometric distribution, 1/1-p = total num of desired appts, solve for p, then random number lower than p means a fu is done

        return patients_to_move_on

def generate_simulation_graph(capacity, resource):
    G = nx.DiGraph()
    G.add_node("Simulate", capacity=capacity, resource=resource)

