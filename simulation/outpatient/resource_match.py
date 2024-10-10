import pandas as pd
import sqlalchemy as sa

from itertools import cycle

class OutpatientResourceMatcher:
    def __init__(self, engine, treatment_function_code):
        # Load the resource information
        resource_df = pd.read_sql(f"EXECUTE [wl].[outpatient_clinics] @tfc='{treatment_function_code}'", engine)

        breakpoint()

        self.resource = self.__process_capacity(resource_df)

    def __process_capacity(self, resource_df):

        resource_df = resource_df.groupby("SessionDate").sum()[["First_nf2f", "First_f2f", "Follow-up_nf2f", "Follow-up_f2f", "unknown", "Total_slots"]]
        resource_df.index = pd.to_datetime(resource_df.index)
        resource_df = resource_df.asfreq('D').fillna(0)

        # start a year ago
        resource_df = resource_df.iloc[-366:-1]

        return cycle(resource_df.iterrows())

    def match_resource(self, wait_list):

        breakpoint()


        # indices = []
        # for appointment types in next(self.resource):
        #  indices += subset wait_list["appointment_type"].iloc[: num of slots].index

        # return indices

        # TODO: deal with FU appointments here
        pass

if __name__ == '__main__':
    engine = sa.create_engine(open(f"op_sql/engine.txt", "r").read())

    oprm = OutpatientResourceMatcher(engine, 110)

    treatment_function_code = 110
    wait_list = pd.read_sql(f"Execute [wl].[current_opa_waiting_list] @site=mph, @tfc='{treatment_function_code}'", engine)

    oprm.match_resource(wait_list)
