from collections import defaultdict

from ..simulation_components import Metrics


class MRIMetrics(Metrics):
    def __init__(self):
        super().__init__()

    def update_metrics(
        self,
        capacity_object,
        num_patients_seen: int,
        num_dnas: int,
        num_cancellations: int,
    ) -> None:
        super().update_metrics(
            capacity_object, num_patients_seen, num_dnas, num_cancellations
        )
        due_day_present = ~capacity_object.wait_list["days_until_due"].isna()

        six_weeks = 6*7
        self.metrics["over_6_weeks_planned"].append((
                (capacity_object.wait_list.loc[due_day_present, "days waited"] -
                 capacity_object.wait_list.loc[due_day_present,"days_until_due"]) > six_weeks
        ).sum())
        self.metrics["over_6_weeks_unplanned"].append((
            (capacity_object.wait_list.loc[~due_day_present,"days waited"] > six_weeks).sum()
        ).sum())

        thirteen_weeks = 13*7
        self.metrics["over_13_weeks_planned"].append((
            (capacity_object.wait_list.loc[due_day_present, "days waited"] -
            capacity_object.wait_list.loc[due_day_present, "days_until_due"]) > thirteen_weeks
        ).sum())
        self.metrics["over_13_weeks_unplanned"].append((
            (capacity_object.wait_list.loc[~due_day_present, "days waited"] > thirteen_weeks).sum()
        ).sum())

        self.metrics["over_6_weeks"].append(
            (capacity_object.wait_list.loc[:, "days waited"] > six_weeks).sum()
        )
