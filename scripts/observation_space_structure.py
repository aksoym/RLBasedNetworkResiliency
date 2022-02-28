import pandas as pd
import numpy as np



arrival_matrices = pd.read_pickle("../data/hourly_arrival_matrix_updated.pickle")
departure_matrices = pd.read_pickle("../data/hourly_departure_matrix_updated.pickle")

def get_random_flight_matrices(arrival_matrices, departure_matrices, n_apt=133, random_idx=None, seed=1713):
    assert len(arrival_matrices) == len(departure_matrices)
    
    if random_idx is None:
        rng = np.random.default_rng(seed=seed)
        random_idx = rng.integers(0, len(arrival_matrices) / n_apt)
    else:
        random_idx = random_idx
    
    def _slicer(idx, hour_increment):
        #This assignment is only for readability.
        step = hour_increment
        return slice((random_idx + step)*n_apt, (random_idx + step + 1)*n_apt)
    
    arr_first = arrival_matrices.iloc[_slicer(random_idx, hour_increment=0), :].values
    arr_second = arrival_matrices.iloc[_slicer(random_idx, hour_increment=1), :].values
    arr_third = arrival_matrices.iloc[_slicer(random_idx, hour_increment=2), :].values
    
    dep_first = departure_matrices.iloc[_slicer(random_idx, hour_increment=0), :].values
    dep_second = departure_matrices.iloc[_slicer(random_idx, hour_increment=1), :].values
    dep_third = departure_matrices.iloc[_slicer(random_idx, hour_increment=2), :].values
    
    return (arr_first, arr_second, arr_third, dep_first, dep_second, dep_third)


def _random_flight_getter_semantic_checker():
    rng = np.random.default_rng(seed=1733)
    flight_flow = pd.read_pickle("../data/all_time_flight_flow_df.pickle")
    random_idx = rng.integers(0, len(flight_flow) / 133)
    matrix_tuple = get_random_flight_matrices(arrival_matrices, departure_matrices, random_idx=random_idx)
    
    ground_truth_flow_matrix = flight_flow.iloc[random_idx*133:(random_idx+1)*133, :].values
    reconstructed_flow_matrix = np.sum(list(matrix_tuple), axis=0)
    
    return np.equal(ground_truth_flow_matrix, reconstructed_flow_matrix)
    
    #ground_truth_flow_matrix, reconstructed_flow_matrix
    
print(np.count_nonzero(_random_flight_getter_semantic_checker()))
    
    
    
    