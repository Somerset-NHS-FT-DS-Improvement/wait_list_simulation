import pandas as pd
import json
from typing import List
    
class MRIDepartment:
    """
    Initialise the Department with resources data from a JSON file.
    
    Args:
        json_file_path (str): Path to the JSON file containing resource data.
    """
    
    def __init__(self, json_file_path: str) -> None:
        with open(json_file_path, 'r') as input_json_file:
            self.resources = json.load(input_json_file)  
            
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
        return int(time_str.split(':')[0]) * 60 + int(time_str.split(':')[1])
    
    
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
    
    
    def __update_metrics(self, time_not_utilised: List[dict], day_num: int) -> None:
        """
        Updates the unutilised resource metrics with the newly calculated unused time for slots.

        Args:
            time_not_utilised (list): List of dictionaries containing details of unused resource time.
            day_num (int): Simulation day number.
        """        
        self.unutilised_resource_metrics[day_num] = time_not_utilised
            
        
    def match_mri_resource(self, waiting_list_df: pd.DataFrame, day: str, day_num: int) -> List[int]:
        """
        Matches patients from the waiting list to available resource slots for a given day.
        Tracks patients who are seen, resource time not utilised.

        Args:
            waiting_list_df (pd.DataFrame): DataFrame containing the patient waiting list.
            day (str): Day of the week for scheduling patients (e.g., '0' for Monday, '1' for Tuesday).
            day_num (int): Simulation day number, used to track the scheduling process across different simulation days.

        Returns:
            list: Indices of patients who were successfully matched and scheduled for slots.

        Additional Details:
            - Unused resource time ('not_utilised_mins') is tracked for each slot.
            - The total unused time across multiple runs is stored in `self.unutilised_resource_metrics` for later analysis.
            - Each resource's daily schedule is matched based on the type of activity and available time slots.
        """
        time_not_utilised = []
        matched_indices = set()
        
        # each resource
        for resource_name, resource_info in self.resources.items():
            # get slots for the given day
            day_slots = resource_info.get('day', {}).get(day)

            if not day_slots:
                # TODO: Add message here when logging is implemented.
                continue

            # filter patients relevant to this resource
            resource_waiting_list_df = waiting_list_df[(waiting_list_df[resource_name] == 1)]
            
            # go through the slots for this resource, day
            for slot in day_slots:
                open_time = self.time_to_minutes(slot['open'])
                close_time = self.time_to_minutes(slot['close'])
                available_duration = close_time - open_time
                label = slot['label']

                # filter the waiting_list by the type of activity for the resource
                activity_matched_list_df = resource_waiting_list_df[resource_waiting_list_df['activity'] == label]
                        
                total_scheduled_time = 0

                # each patient in the matched activity slot for this resource, day
                for index, patient in activity_matched_list_df.iterrows():
                    duration = patient['duration_mins']
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
                
                time_not_utilised.append({
                    'resource_name': resource_name,
                    'day': day,
                    'slot': slot,
                    'not_utilised_mins': available_duration - total_scheduled_time,
                    'activity': label
                })   
        
        self.__update_metrics(time_not_utilised, day_num)
        
        # TODO: Follow-ups
        
        return matched_indices