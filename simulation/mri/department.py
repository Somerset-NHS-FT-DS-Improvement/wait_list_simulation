import pandas as pd
import json

    
class Department:
    """
    Initialise the Department with resources data from a JSON file.
    
    Parameters:
        json_file_path (str): Path to the JSON file containing resource data.
    """
    
    def __init__(self, json_file_path):
        with open(json_file_path, 'r') as input_json_file:
            self.resources = json.load(input_json_file)    
            
    @staticmethod
    def time_to_minutes(time_str):
        """
        Converts a time string in 'HH:MM' format to total minutes.
        
        Parameters:
            time_str (str): Time string in 'HH:MM' format.
        
        Returns:
            int: Total minutes.
        """
        return int(time_str.split(':')[0]) * 60 + int(time_str.split(':')[1])
    
    @staticmethod
    def minutes_to_time(minutes):
        """
        Converts total minutes to a time string in 'HH:MM' format.
        
        Parameters:
            minutes (int): Total minutes.
        
        Returns:
            str: Time string in 'HH:MM' format.
        """
        return f"{minutes // 60:02d}:{minutes % 60:02d}"
    
    def match_mri_resource(self, waiting_list_df, day, day_num): 
        """
        Matches patients from the waiting list to available resource slots for a given day.
        Tracks patients who are seen, resource time not utilised.

        Parameters:
            waiting_list_df (pd.DataFrame): DataFrame containing the patient waiting list.
            day (str): Day of the week for scheduling patients (e.g., '0' for Monday, '1' for Tuesday).
            day_num (int): Simulation day number, used to track the scheduling process across different simulation days.
        
        Returns:
            list: Indices of patients who were successfully matched and scheduled for slots.
        
        Additional Details:
            - Unused resource time ('not_utilised_mins') is tracked for each slot.
            - The total unused time across multiple runs is stored in `self.wasted_resource_metrics` for later analysis.
            - Each resource's daily schedule is matched based on the type of activity and available time slots.
        """
        patients_seen = []
        time_not_utilised = []
        waiting_list_df.reset_index(drop=True, inplace=True)
        matched_indices = []
        
        # each resource
        for resource_name, resource_info in self.resources.items():
            # get slots for the given day
            day_slots = resource_info.get('day', {}).get(day)

            if not day_slots:
                # print(f"No available slots for resource {resource_name} on day {day}. Skipping.")
                continue

            # filter patients relevant to this resource
            resource_waiting_list_df = waiting_list_df[(waiting_list_df[resource_name] == 1)]
            extra_scheduled_idx = None
            
            # go through the slots for this resource, day
            for slot in day_slots:
                open_time = self.time_to_minutes(slot['open'])
                close_time = self.time_to_minutes(slot['close'])
                available_duration = close_time - open_time
                label = slot['label']

                # filter the waiting_list by the type of activity for the resource
                activity_matched_list_df = resource_waiting_list_df[resource_waiting_list_df['activity'] == label]
                          
                total_scheduled_time = 0
                check_index = None

                # each patient in the matched activity slot for this resource, day
                for index, patient in activity_matched_list_df.iterrows():
                    duration = patient['duration_mins']
                    check_index = index
     
                    # check patient can be scheduled within available slot time
                    if total_scheduled_time + duration <= available_duration:
                        # patient seen
                        
                        # patients_seen.append({
                        #     'patient_id': patient['mrn'],
                        #     'resource': resource_name,
                        #     'day': day,
                        #     'day_num': day_num,
                        #     'slot': slot,
                        #     'scheduled_time': self.minutes_to_time(open_time + int(total_scheduled_time)),
                        #     'duration_mins': patient['duration_mins'],
                        #     'activity': label
                        # })
                        
                        total_scheduled_time += duration
                        matched_indices.append(index)
                        
                        # remove the patient from dataframes
                        resource_waiting_list_df = resource_waiting_list_df.drop(index)
                        waiting_list_df = waiting_list_df.drop(index)
                        activity_matched_list_df = resource_waiting_list_df[resource_waiting_list_df['activity'] == label]
                        
                    else:
                        remaining_time = available_duration - total_scheduled_time
                        if remaining_time >= 15:
                            counter = 0
                            for next_idx, next_patient in activity_matched_list_df.iterrows():
                                if  counter > 2: # stop after checking next TWO patients
                                    break
                        
                                counter += 1
                                
                                if next_idx != check_index and next_patient['duration_mins'] <= remaining_time:
                                    # schedule patient
                                    extra_scheduled_idx = next_idx
                                    
                                    # patients_seen.append({
                                    #     'patient_id': next_patient['mrn'],
                                    #     'resource': resource_name,
                                    #     'day': day,
                                    #     'day_num': day_num,
                                    #     'slot': slot,
                                    #     'scheduled_time': self.minutes_to_time(open_time + int(total_scheduled_time)),
                                    #     'duration_mins': next_patient['duration_mins'],
                                    #     'activity': label
                                    # })
                    
                                    total_scheduled_time += next_patient['duration_mins']
                                    matched_indices.append(next_idx)
                                    break
                        
                        # time slot exceeded, break and move to next slot
                        time_not_utilised.append({
                            'resource_name': resource_name,
                            'day': day,
                            'day_num': day_num,
                            'slot':slot,
                            'not_utilised_mins': available_duration - total_scheduled_time,
                            'activity': label
                        }) 
                        
                        if extra_scheduled_idx != None:                            
                            resource_waiting_list_df = resource_waiting_list_df.drop(extra_scheduled_idx)
                            waiting_list_df = waiting_list_df.drop(extra_scheduled_idx) 
                            extra_scheduled_idx = None           
                                    
                        break  

        time_not_utilised_df = pd.DataFrame(time_not_utilised)
        
        if hasattr(self, 'wasted_resource_metrics'):
            self.wasted_resource_metrics = pd.concat([self.wasted_resource_metrics, time_not_utilised_df], ignore_index=True)
        else:
            self.wasted_resource_metrics = time_not_utilised_df
        
        return matched_indices

