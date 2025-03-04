import json
from typing import Any, Dict, List, Tuple

import pandas as pd
from numpy.random import Generator


class MRIDepartment:
    """
    Initialise the Department with resources data from a JSON file.

    Args:
        json_file_path (str): Path to the JSON file containing resource data.
        fu_rate (float): The follow-up rate for patients.
        fu_rng (np.random.Generator): A random number generator for follow-up calculations.
        clinic_utilisation (float): Clinic utilisation rate. Default is 1.
    """

    def __init__(
        self,
        json_file_path: str,
        fu_rate: float,
        fu_rng: Generator,
        clinic_utilisation: float = 1,
    ) -> None:
        with open(json_file_path, "r") as input_json_file:
            self.resources = json.load(input_json_file)

        self.fu_rate = fu_rate
        self.clinic_utilisation = clinic_utilisation  # Should this be a distribution?
        self.fu_rng = fu_rng

        self.unutilised_resource_metrics = {}

    @staticmethod
    def time_to_minutes(time_str: str) -> int:
        """
        Converts a time string in 'HH:MM' format to total minutes.

        Args:
            time_str (str): Time string in 'HH:MM' format.

        Returns:
            int: Total minutes.
        """
        return int(time_str.split(":")[0]) * 60 + int(time_str.split(":")[1])

    @staticmethod
    def minutes_to_time(minutes: int) -> str:
        """
        Converts total minutes to a time string in 'HH:MM' format.

        Parameters:
            minutes (int): Total minutes.

        Returns:
            str: Time string in 'HH:MM' format.
        """
        return f"{minutes // 60:02d}:{minutes % 60:02d}"

    def __update_metrics(
        self, time_not_utilised: List[Dict[str, Any]], day_num: int
    ) -> None:
        """
        Updates the unutilised resource metrics with the newly calculated unused time for slots.

        Args:
            time_not_utilised (List[Dict[str, Any]]): List of dictionaries containing details of unused resource time.
            day_num (int): Simulation day number.
        """
        self.unutilised_resource_metrics[day_num] = time_not_utilised

    def match_mri_resource(
        self, waiting_list_df: pd.DataFrame, day: str, day_num: int
    ) -> Tuple[List[int], pd.DataFrame]:
        """
        Matches patients from the waiting list to available resource slots for a given day.
        Tracks patients who are seen, resource time not utilised.

        Args:
            waiting_list_df (pd.DataFrame): DataFrame containing the patient waiting list.
            day (str): Day of the week for scheduling patients (e.g., '0' for Monday, '1' for Tuesday).
            day_num (int): Simulation day number, used to track the scheduling process across different simulation days.

        Returns:
            Tuple[List[int], pd.DataFrame]: Indices of patients who were successfully matched and scheduled for slots,
            and DataFrame of follow-up patients.
        """
        time_not_utilised = []
        matched_indices = set()

        for resource_name, resource_info in self.resources.items():
            day_slots = resource_info.get("day", {}).get(day)

            if not day_slots:
                # TODO: Add message here when logging is implemented.
                continue

            # check if the resource supports pediatric scans
            supports_paediatric_scans = resource_info.get("peadiatric_scans", False)

            # filter patients relevant to this resource
            resource_waiting_list_df = waiting_list_df[
                (waiting_list_df[resource_name] == 1)  # compatible with this resource
                & (
                    (supports_paediatric_scans) | (waiting_list_df["age"] >= 18)
                )  # paediatric compatibility logic
            ]

            # go through the slots for this resource, day
            for slot in day_slots:
                open_time = self.time_to_minutes(slot["open"])
                close_time = self.time_to_minutes(slot["close"])
                available_duration = (close_time - open_time) * self.clinic_utilisation

                label = slot["label"]

                # filter the waiting_list by the type of activity for the resource
                activity_matched_list_df = resource_waiting_list_df[
                    resource_waiting_list_df["activity"] == label
                ]

                total_scheduled_time = 0

                # each patient in the matched activity slot for this resource, day
                for index, patient in activity_matched_list_df.iterrows():
                    duration = patient["duration_mins"]
                    remaining_time = available_duration - total_scheduled_time
                    time_available = remaining_time - duration >= 0

                    if remaining_time < 15:
                        break  # time slot exceeded, break and move to next slot
                    elif index in matched_indices or not time_available:
                        continue
                    else:
                        # patient seen
                        total_scheduled_time += duration
                        matched_indices.add(index)

                time_not_utilised.append(
                    {
                        "resource_name": resource_name,
                        "day": day,
                        "slot": slot,
                        "not_utilised_mins": available_duration - total_scheduled_time,
                        "activity": label,
                    }
                )

        self.__update_metrics(time_not_utilised, day_num)

        fu_df = None
        matched_indices = list(matched_indices)
        if self.fu_rate:
            num_fus = int(len(matched_indices) * self.fu_rate)
            fu_indices = self.fu_rng.choice(matched_indices, num_fus)
            fu_df = waiting_list_df.iloc[fu_indices]
            fu_df["priority"] = "Follow up"
            fu_df["days waited"] = 0

        return matched_indices, fu_df
