from itertools import cycle

import pandas as pd


class OutpatientResourceMatcher:
    def __init__(self, resource_df, fu_rate=0, fu_rng=None):
        self.resource = self.__process_capacity(resource_df)

        self.fu_rate = fu_rate
        self.fu_rng = fu_rng

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
        _, num_appts = next(self.resource)

        indices = (
            wait_list[wait_list["appointment_type"] == "first"]
            .iloc[: num_appts["first"]]
            .index.to_list()
        )
        indices += (
            wait_list[wait_list["appointment_type"] == "followup"]
            .iloc[: num_appts["followup"]]
            .index.to_list()
        )

        indices += (
            wait_list[~wait_list.index.isin(indices)]
            .iloc[: int(num_appts["unknown"])]
            .index.to_list()
        )

        fu_df = None
        if self.fu_rate:
            num_fus = int(len(indices) * self.fu_rate)
            fu_indices = self.fu_rng.choice(indices, num_fus, replace=False)
            fu_df = wait_list.iloc[fu_indices]
            fu_df["priority"] = "Follow-up"
            fu_df["appointment_type"] = "followup"
            fu_df["days waited"] = 0

        return indices, fu_df

