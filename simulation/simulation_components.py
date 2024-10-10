import networkx as nx
import numpy as np
import pandas as pd


class PatientGenerator:
    def __init__(
        self, generate_new_patients, start_id=0
    ):
        self.id = start_id
        self.generate_new_patients = generate_new_patients

    def generate_patients(self, day_num, day):
        # Emergency and new patients dealt with through the interface with generate_new_patients
        patients = self.generate_new_patients(day_num)

        patients.loc[:, "pathway"] = [[] for _ in range(patients.shape[0])]

        patients.loc[:, "id"] = [*range(self.id, self.id + len(patients))]
        self.id += len(patients) - 1

        # Generator object
        iterable_object = patients.iterrows()
        return self.__yield_patient(iterable_object)

    def __yield_patient(self, iterable_object):
        for _, patient in iterable_object:
            yield patient

class EmergencyPatientGenerator:
    def __init__(self):
        pass


def get_appointment_duration(patient):
    return patient.duration_mins


class Capacity:
    def __init__(
        self,
        match_resource,
        prioritisation_calculator,
        dna_rate,
        cancellation_rate,
        fu_rate,
        initial_wait_list,
        rott_removals,
        seed = None,
    ):
        self.wait_list = initial_wait_list
        self.prioritisation_calculator = prioritisation_calculator
        self.match_resource = match_resource

        self.dna_rate = dna_rate
        self.cancellation_rate = cancellation_rate
        self.fu_rate = fu_rate

        self.rott_removals = rott_removals

        self.rng = np.random.default_rng(seed)

        self.wait_list["id"] = [*range(0, self.wait_list.shape[0])]
        self.wait_list["pathway"] = [[] for _ in range(self.wait_list.shape[0])]

        self.metrics = {"maximum_wait_time": [], "wait_list_length": []}

    def get(self, resource, patient, day_num, day):
        self.wait_list = pd.concat([self.wait_list, patient.to_frame().T])
        self.wait_list.reset_index(drop=True, inplace=True)

        return True

    def update_day(self, day_num, day):
        # prioritise wait list
        indices = self.prioritisation_calculator.calculate_sorted_indices(
            self.wait_list
        )
        self.wait_list = self.wait_list.iloc[indices]

        patients_to_move_on_indices = self.match_resource(self.wait_list, day, day_num)

        patients_to_move_on = self.wait_list[self.wait_list.index.isin(patients_to_move_on_indices)]
        self.wait_list.drop(index=patients_to_move_on_indices, inplace=True)

        # TODO: add patients to have fu to the wait list with new FU label -- Do this in the resource, send back the dataframe of FUs

        # TODO: Cancel and DNA needs to be taken from the patients to move on -- check the process for DNAs and cancellations )do they go back on the list?)

        # Or take patients to move on, and randomly select to have FU appts?
        # Use a geometric distribution, 1/1-p = total num of desired appts, solve for p, then random number lower than p means a fu is done

        rott_patients = self.rng.choice(self.wait_list.index, self.rott_removals[day_num])
        self.wait_list.drop(index=rott_patients, inplace=True)

        self.__update_metrics()
        self.wait_list["days waited"] += 1

        iterable_patients = patients_to_move_on.iterrows()
        return self.__yield_patient(iterable_patients)

    def __update_metrics(self):
        self.metrics["maximum_wait_time"].append(self.wait_list["days waited"].max())
        self.metrics["wait_list_length"].append(self.wait_list.shape[0])

    def __yield_patient(self, iterable_object):
        for _, patient in iterable_object:
            yield patient


def generate_simulation_graph(capacity, resource):
    G = nx.DiGraph()
    G.add_node("Simulate", capacity=capacity, resource=resource)
    return G
