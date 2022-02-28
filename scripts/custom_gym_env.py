from unittest import result
import gym
from gym import spaces
import numpy as np
import pandas as pd

from utils import fetch_single_flow_matrix, fetch_hourly_flow_matrices

flight_flows = pd.read_pickle("../data/all_time_flight_flow_df.pickle")
arr_flows = pd.read_pickle("../data/hourly_arrival_matrix_updated.pickle")
dep_flows = pd.read_pickle("../data/hourly_departure_matrix_updated.pickle")

rng = np.random.default_rng(seed=1455)

class AirTrafficFlow(gym.Env):
    
    NUM_OF_OPERATIONS = 4 #0-> do nothing, 1-> delay 1hr, 2-> delay 2h, 3-> cancel flight.
    
    def __init__(self, n_apt: int) -> None:
        super().__init__()
        
        self.n_apt = n_apt
        self.action_space = spaces.MultiDiscrete([n_apt*(n_apt - 1), self.NUM_OF_OPERATIONS])
        
        self.observation_space = spaces.Tuple(
            (
                spaces.Box(low=-np.inf, high=np.inf, shape=(n_apt, n_apt), dtype=np.float16),
                spaces.Box(low=-np.inf, high=np.inf, shape=(n_apt, n_apt), dtype=np.float16),
                spaces.Box(low=-np.inf, high=np.inf, shape=(n_apt, n_apt), dtype=np.float16)
            )
            )
        
        
    def step(self, action):
        #Parse the action.
        entry_index, operation = action
        #Locate the entry index on the flow matrix.
        #First skip diagonal indices, because they represent self loops on flow matrix.
        diagonal = np.diag_indices(self.n_apt, ndim=2)
        #If entry_index corresponds to a diagonal index, skip to next one.
        if entry_index in diagonal:
            entry_index += 1
        
        #Operation index.
        if operation == 0:
            pass
        elif operation == 1:
            first_hour_departure = self.hourly_matrix_list[3]
            second_hour_departure = self.hourly_matrix_list[4]
            row, column = divmod(entry_index, self.n_apt)
            first_hour_departure.iloc[row, column] -= 1
            second_hour_departure.iloc[row, column] += 1
            
        elif operation == 2:
            first_hour_departure = self.hourly_matrix_list[3]
            third_hour_departure = self.hourly_matrix_list[5]
            row, column = divmod(entry_index, self.n_apt)
            first_hour_departure.iloc[row, column] -= 1
            third_hour_departure.iloc[row, column] += 1
            
        elif operation == 3:
            first_hour_departure = self.hourly_matrix_list[3]
            row, column = divmod(entry_index, self.n_apt)
            first_hour_departure.iloc[row, column] -= 1
        else:
            raise ValueError("Opeartion action must be one of the (0, 1, 2, 3)")
            
        resulting_observation = _combine_hourly_flows_and_get_stability_matrices(self.hourly_matrix_list, 
                                                                                     self.recovery_rates)
        reward_from_observation = _reward(resulting_observation)
        
            
        
        
            
    def reset(self):
        #Get the hourly matrix list. [arr1, arr2, arr3, dep1, dep2, dep3]
        self.hourly_matrix_list = fetch_hourly_flow_matrices(arr_flows, dep_flows, random=True)
        
        #Initialize a random recovery rate vector.
        self.recovery_rates = rng.normal(loc=1, scale=2, size=(133,))
        
        self.stability_matrices = _combine_hourly_flows_and_get_stability_matrices(self.hourly_matrix_list, 
                                                                                   self.recovery_rates)
        
        return self.stability_matrices
        
    def _combine_hourly_flows_and_get_stability_matrices(hourly_matrix_list, recovery_rates):
        #Combine departure and arrivals together.
        combined_flow_matrices = [hourly_matrix_list[idx] + hourly_matrix_list[idx+3] 
                                       for idx in range(3)]
        
        infection_matrices = []
        for matrix_df in combined_flow_matrices:
            row_sums = matrix_df.sum(axis=1)
            #Divide each row with the sum and convert the result to an ndarray with .values method.
            infection_matrix = matrix_df.div(row_sums, axis=0).values
            infection_matrix = np.nan_to_num(infection_matrix, copy=True, nan=0.0)
            infection_matrices.append(infection_matrix)
        
        #After filling the diagonal, infection matrix becomes the stability matrix, this line is for readability.
        stability_matrices = []
        for matrix in infection_matrices:
            
            np.fill_diagonal(matrix, recovery_rates)
            matrix = matrix.astype(np.float16)
            
            stability_matrices.append(matrix)
        
        return tuple(stability_matrices)
        
        
env = AirTrafficFlow(133)
print(env.reset())


        
       
    