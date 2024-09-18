import pandas as pd
import numpy as np
from collections import Counter


def fetch_historic_sql():

    """
    !ToDo
    Function containing sql to historical data for proportions.
    """
    pass

def patient_categoriser(fcst_additions, hierarchical_cols, historic_data, seed=42):

    """
    Place patients in categories based on historic proportions.

    Args:
        fcst_additions (Pandas.Series): Future forecast of total additions (output from forecast class).
        hierarchical_cols (list): Needs to be ordered by set size (i.e., priority->procedure...).
        historic_data (Pandas.DataFrame): Must contain columns from hierarchical_cols.

    Returns:
        Pandas.DataFrame: A dictionary with hierarchical categories as keys and patient counts as values.
    """

    np.random.seed(seed)

    cleaned_data = historic_data.dropna(subset=hierarchical_cols, how='any')

    grouped = cleaned_data.groupby(by=hierarchical_cols).size()
    total_count = len(cleaned_data)
    probabilities = grouped / total_count

    probabilities_df = pd.DataFrame(probabilities, columns=['proba']).reset_index()

    print(probabilities_df)

    categoriser_list=[]
    for f in fcst_additions.values:

        prob_col = probabilities_df.groupby(by = hierarchical_cols).sum(numeric_only=1)['proba']
        forecast_count = np.random.choice(prob_col.index,
                                          size=f,
                                          p=prob_col)

        categoriser_list.append(dict(Counter(forecast_count)))

    output_categoriser = pd.DataFrame()
    output_categoriser['ds'] = fcst_additions.index
    output_categoriser['cat'] = categoriser_list

    return output_categoriser.set_index('ds')