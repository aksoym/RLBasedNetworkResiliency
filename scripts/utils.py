import pandas as pd
import numpy as np
from typing import List

#flight_flows = pd.read_pickle("../data/all_time_flight_flow_df.pickle")
#TODO embed rng function into single flow matrix func.

def fetch_single_flow_matrix(flight_flow_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    #Either locate the matrix by an integer indexer or by date tw pair.
    if "index" in kwargs:
        idx = kwargs["index"]
        apt_count = len(flight_flow_df.columns)
        return flight_flow_df.iloc[idx*apt_count:(idx+1)*apt_count, :]
    
    elif "random" in kwargs and kwargs["random"] is True:
        apt_count = len(flight_flow_df.columns)
        #not checking for float error because len(flight_flows)/apt_count is int/int and cannot be any other form.
        #because flow_matrix is created by the multiples of apt_count.
        idx = int(rng.integers(0, len(flight_flow_df)/apt_count, size=1))
        return flight_flow_df.iloc[idx*apt_count:(idx+1)*apt_count, :]
    
    else:
        date = kwargs["date"]
        tw = kwargs["tw"]
        return flight_flow_df.loc[(date, tw), :]


def fetch_hourly_flow_matrices(arr_flow, dep_flow, n_apt, rng, random=True, **kwargs):
    
    if random:
        #Minus three because we will increment the idx already.
        idx = int(rng.integers(0, (len(arr_flow)/n_apt) - 3, size=1))
        arr_first = arr_flow.iloc[idx*n_apt:(idx+1)*n_apt, :]
        dep_first = dep_flow.iloc[idx*n_apt:(idx+1)*n_apt, :]
        
        idx += 1
        arr_second = arr_flow.iloc[idx*n_apt:(idx+1)*n_apt, :]
        dep_second = dep_flow.iloc[idx*n_apt:(idx+1)*n_apt, :]
        
        idx += 1
        arr_third = arr_flow.iloc[idx*n_apt:(idx+1)*n_apt, :]
        dep_third = dep_flow.iloc[idx*n_apt:(idx+1)*n_apt, :]
        
        df_list = [arr_first, arr_second, arr_third, dep_first, dep_second, dep_third]
        
        return _crop_dataframes(df_list, n_apt)
    else:
        assert "index" in kwargs, "'index' must be passed if random is False"
        idx = kwargs["index"]
        arr_first = arr_flow.iloc[idx*n_apt:(idx+1)*n_apt, :]
        dep_first = dep_flow.iloc[idx*n_apt:(idx+1)*n_apt, :]
        
        idx += 1
        arr_second = arr_flow.iloc[idx*n_apt:(idx+1)*n_apt, :]
        dep_second = dep_flow.iloc[idx*n_apt:(idx+1)*n_apt, :]
        
        idx += 1
        arr_third = arr_flow.iloc[idx*n_apt:(idx+1)*n_apt, :]
        dep_third = dep_flow.iloc[idx*n_apt:(idx+1)*n_apt, :]
        
        df_list = [arr_first, arr_second, arr_third, dep_first, dep_second, dep_third]
        
        return _crop_dataframes(df_list, n_apt)


def _crop_dataframes(dfs: List[pd.DataFrame], size_after_crop: int) -> List[pd.DataFrame]:
    df_list = []
    for df in dfs:
        df_list.append(df.iloc[0:size_after_crop, 0:size_after_crop])
    return df_list
        
        
    
    
