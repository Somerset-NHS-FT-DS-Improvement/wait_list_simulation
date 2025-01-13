import pandas as pd
import sqlalchemy as sa


class Data:
    def __init__(
        self,
        path_to_sql_queries,
        historic_waiting_list_file_name,
        num_new_refs_file_name,
        current_waiting_list_file_name,
        fu_file_name=None,
        dna_file_name=None,
        cancellation_file_name=None,
        rott_file_name=None,
    ):
        self.engine = sa.create_engine(
            open(f"{path_to_sql_queries}/engine.txt", "r").read()
        )

        self.fu_rate = None
        self.dna_rate = None
        self.cancellation_rate = None

        if fu_file_name is not None:
            self.fu_rate = pd.read_sql(
                open(f"{path_to_sql_queries}/{fu_file_name}", "r").read(),
                self.engine,
            ).values[0, 0]

        if dna_file_name is not None:
            self.dna_rate = pd.read_sql(
                open(f"{path_to_sql_queries}/{dna_file_name}", "r").read(), self.engine
            ).values[0, 0]

        if cancellation_file_name is not None:
            self.cancellation_rate = pd.read_sql(
                open(f"{path_to_sql_queries}/{cancellation_file_name}", "r").read(),
                self.engine,
            ).values[0, 0]

        if rott_file_name is not None:
            self.rott = pd.read_sql(
                open(f"{path_to_sql_queries}/{rott_file_name}", "r").read(),
                self.engine,
            )

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
