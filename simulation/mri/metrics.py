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
        self.metrics["num_6_plus"].append(
            (capacity_object.wait_list["days waited"] >= (7 * 6)).sum()
        )
        self.metrics["num_13_plus"].append(
            (capacity_object.wait_list["days waited"] >= (7 * 13)).sum()
        )
