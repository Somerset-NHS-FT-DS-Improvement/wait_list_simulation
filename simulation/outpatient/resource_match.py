from itertools import cycle

import pandas as pd
import numpy as np


class OutpatientResourceMatcher:
    def __init__(self, resource_df, fu_rate=0, fu_rng=None):
        self.resource = self.__process_capacity(resource_df)

        self.fu_rate = fu_rate
        self.fu_rng = fu_rng

        self.borrowed_capacity = {"first":0, "followup":0}
        self.unutilised_appts = {}
        self.borrowed_capacities = {}

    def __process_capacity(self, resource_df):
        resource_df = resource_df.groupby("SessionDate").sum()[
            ["first", "followup", "unknown", "Total_slots"]
        ]
        resource_df.index = pd.to_datetime(resource_df.index)
        resource_df = resource_df.asfreq("D").fillna(0)
        resource_df = resource_df.astype(int)

        # start a year ago
        resource_df = resource_df.iloc[-366:-1]

        return cycle(resource_df.iterrows())

    def match_resource(self, wait_list, day, day_num):
        must_be_seen = wait_list[wait_list["sim_day_appt_due"] == day_num]

        _, num_appts = next(self.resource)

        indices = []

        for appt_type in ["first", "followup"]:
            if self.borrowed_capacity[appt_type] != 0:
                num_appts[appt_type] += self.borrowed_capacity[appt_type]
                self.borrowed_capacity[appt_type] = 0

            tmp_indices = must_be_seen[must_be_seen["appointment_type"] == appt_type].index.to_list()
            remaining_appts = num_appts[appt_type] - len(tmp_indices)
            if remaining_appts < 0:
                num_appts[appt_type] = 0
                self.borrowed_capacity[appt_type] += remaining_appts
            else:
                num_appts[appt_type] = remaining_appts

            indices += tmp_indices

            indices += (
                wait_list[(wait_list["appointment_type"] == appt_type) & (wait_list["sim_day_appt_due"].isna())]
                .iloc[: num_appts[appt_type]]
                .index.to_list()
            )
        self._update_metrics(day_num, num_appts)

        fu_df = None
        if self.fu_rate:
            num_fus = int(len(indices) * self.fu_rate)
            fu_indices = self.fu_rng.choice(indices, num_fus, replace=False)
            fu_df = wait_list.iloc[fu_indices]
            fu_df["priority"] = "Follow-up"
            fu_df["appointment_type"] = "followup"
            fu_df["min_wait"] = 21
            fu_df["max_wait"] = 126
            fu_df["days waited"] = 0
            fu_df["sim_day_appt_due"] = np.nan

        return indices, fu_df


    def _update_metrics(self, day_num, num_appts):
        self.unutilised_appts[day_num] = num_appts
        self.borrowed_capacities[day_num] = self.borrowed_capacity.copy()

