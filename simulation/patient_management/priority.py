import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


class PriorityCalculator:
    def __init__(self, priority_order: List[str]):
        """
        Initialise the PriorityCalculator with a priority order.

        Parameters:
        priority_order (List[str]): A list of column names in the order of priority for sorting.
        """
        self.priority_order = priority_order

    def calculate_sorted_indices(self, df: pd.DataFrame) -> np.ndarray:
        """
        Validate the DataFrame and calculate sorted indices based on priority.

        This function checks that the required columns 'priority', 'setting', and 'days waited'
        are present in the DataFrame. It then calculates the minimum and maximum wait times and
        sorts the DataFrame based on the priority order.

        Parameters:
        df (pd.DataFrame): The input DataFrame to be validated and processed.

        Returns:
        np.ndarray: An array of sorted indices based on the priority order.

        Raises:
        AssertionError: If the required columns 'priority', 'setting', and 'days waited' are not present in the DataFrame.
        """
        assert (
            len(set(df.columns) - {"priority", "setting", "days waited"})
            == len(df.columns) - 3
        ), f"The columns priority, setting and days waited are required and not found in {df.columns}"

        min_max_wait_times = self.calculate_min_and_max_wait_times(df)

        priority_mapping = self.__get_priority_mapping(df, min_max_wait_times)
        sorted_indices = self.calculate_wait_list_order(priority_mapping)

        return sorted_indices

    def calculate_min_and_max_wait_times(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate the minimum and maximum wait times for each entry in the DataFrame based on priority.

        This function applies a regex mapping to determine the minimum and maximum wait times for each entry
        in the 'priority' column of the DataFrame and assigns these times to a numpy array.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing the 'priority' column.

        Returns:
        np.ndarray: A 2D numpy array where each row contains the [MinWaitTime, MaxWaitTime] for the corresponding entry.
        """
        regex_mapping_min_max_wait = self.__get_regex_mapping()

        min_max_wait_times = np.array(
            [
                *zip(
                    *df["priority"].apply(
                        self.apply_regex_map,
                        regex_mapping=regex_mapping_min_max_wait,
                        default_value=[21, 126],
                    )
                )
            ]
        ).T

        return min_max_wait_times

    def __get_regex_mapping(self) -> Dict[str, List[int]]:
        """
        Load the regex mapping for minimum and maximum wait times from a JSON file.

        Returns:
        Dict[str, List[int]]: A dictionary where keys are regex patterns and values are lists of min and max wait times.
        """
        config_file = (
            Path(__file__).resolve().parent.parent
            / "config"
            / "min_max_wait_mapping.json"
        )
        with open(config_file, "r") as fin:
            return json.load(fin)

    def apply_regex_map(
        self, val: str, regex_mapping: Dict[str, List[int]], default_value: List[int]
    ) -> List[int]:
        """
        Map a value to specific wait times using regular expressions.

        This function takes a value, a mapping dictionary with regular expressions as keys, and associated wait times
        as values. It attempts to find a regular expression match for the given value and returns the corresponding wait time.
        If no match is found, it returns the default value.

        Parameters:
        val (str): The value to be matched using regular expressions.
        regex_mapping (Dict[str, List[int]]): A dictionary with regular expressions as keys and associated wait times as values.
        default_value (List[int]): The value to be returned if no match is found.

        Returns:
        List[int]: The wait time associated with the matched regular expression, or the default value if no match is found.

        Raises:
        ValueError: If no regular expression in the map matches the provided value.
        """
        if not val:
            return default_value
        for reg_map, wait_times in regex_mapping.items():
            if re.search(reg_map, str(val)):
                return wait_times
        raise ValueError(f"No match found for {val}")

    def calculate_wait_list_order(
        self, priority_mapping: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Calculate the order of the waitlist based on the given priority order.

        This function applies lexicographical sorting based on the provided priority order,
        where the final sorting criterion is applied first.

        Parameters:
        priority_mapping (Dict[str, np.ndarray]): A dictionary where keys are priority criteria and values are arrays used for sorting.

        Returns:
        np.ndarray: An array of sorted indices based on the priority order.
        """
        return np.lexsort(
            [priority_mapping[priority] for priority in self.priority_order[::-1]]
        )  # reverse the order so that the final sort is the last one applied

    @staticmethod
    def __get_priority_mapping(
        df: pd.DataFrame, min_max_wait_times: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Create a priority mapping for sorting based on specific conditions.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing the necessary columns for priority calculation.
        min_max_wait_times (np.ndarray): A 2D numpy array where each row contains the [MinWaitTime, MaxWaitTime] for each entry.

        Returns:
        Dict[str, np.ndarray]: A dictionary where keys are priority criteria and values are arrays used for sorting.
        """
        return {
            "A&E patients": ~(df["setting"] == "A&E Patient"),
            "inpatients": ~(df["setting"] == "Inpatient"),
            "Breach": -(df["days waited"] - min_max_wait_times[:, 1]),
            "Days waited": -min_max_wait_times[:, 1],
            "Over minimum wait time": ~(
                df["days waited"] > min_max_wait_times[:, 0]
            ),  # The inversion here because False will be sorted before True (0 comes before 1)
            "Under maximum wait time": ~(df["days waited"] > min_max_wait_times[:, 1]),
        }
