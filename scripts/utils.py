import pandas as pd
import numpy as np

flight_flows = pd.read_pickle("../data/all_time_flight_flow_df.pickle")
rng = np.random.default_rng(seed=1130)

def fetch_single_flow_matrix(flight_flows: pd.DataFrame, **kwargs) -> pd.DataFrame:
    #Either locate the matrix by an integer indexer or by date tw pair.
    if "index" in kwargs:
        idx = kwargs["index"]
        apt_count = len(flight_flows.columns)
        return flight_flows.iloc[idx*apt_count:(idx+1)*apt_count, :]
    
    elif "random" in kwargs and kwargs["random"] is True:
        apt_count = len(flight_flows.columns)
        #not checking for float error because len(flight_flows)/apt_count is int/int and cannot be any other form.
        #because flow_matrix is created by the multiples of apt_count.
        idx = int(rng.integers(0, len(flight_flows)/apt_count, size=1))
        return flight_flows.iloc[idx*apt_count:(idx+1)*apt_count, :]
    
    else:
        date = kwargs["date"]
        tw = kwargs["tw"]
        return flight_flows.loc[(date, tw), :]

