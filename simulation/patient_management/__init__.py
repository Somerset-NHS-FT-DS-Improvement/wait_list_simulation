import sqlalchemy as sa
import pandas as pd


class Data:
    def __init__(
        self,
        path_to_sql_queries,
        historic_waiting_list_file_name,
        num_new_refs_file_name,
        current_waiting_list_file_name,
        dna_file_name=None,
        cancellation_file_name=None,
    ):
        self.engine = sa.create_engine(
            open(f"{path_to_sql_queries}/engine.txt", "r").read()
        )

        self.dna_rate = None
        self.cancellation_rate = None

        if dna_file_name is not None:
            self.dna_rate = pd.read_sql(
                open(f"{path_to_sql_queries}/{dna_file_name}", "r").read(), self.engine
            ).values[0, 0]

        if cancellation_file_name is not None:
            self.cancellation_rate = pd.read_sql(
                open(f"{path_to_sql_queries}/{cancellation_file_name}", "r").read(),
                self.engine,
            ).values[0, 0]

        # TODO FU rate, ROTT params

        self.historic_waiting_list = pd.read_sql(
            open(
                f"{path_to_sql_queries}/{historic_waiting_list_file_name}", "r"
            ).read(),
            self.engine,
        )

        self.num_new_refs = pd.read_sql(
            open(f"{path_to_sql_queries}/{num_new_refs_file_name}", "r").read(),
            self.engine,
        )

        self.current_waiting_list = pd.read_sql(
            open(f"{path_to_sql_queries}/{current_waiting_list_file_name}", "r").read(),
            self.engine,
        )
