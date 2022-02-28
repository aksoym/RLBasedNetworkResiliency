from tabnanny import check
from unittest import result
import gym
from gym import spaces
import numpy as np
import pandas as pd
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.ppo.ppo import PPO

from utils import fetch_single_flow_matrix, fetch_hourly_flow_matrices

pd.options.mode.chained_assignment = None

flight_flows = pd.read_pickle("../data/all_time_flight_flow_df.pickle")
arr_flows = pd.read_pickle("../data/hourly_arrival_matrix_updated.pickle")
dep_flows = pd.read_pickle("../data/hourly_departure_matrix_updated.pickle")

class AirTrafficFlow(gym.Env):
    
    
    NUM_OF_OPERATIONS = 4 #0-> do nothing, 1-> delay 1hr, 2-> delay 2h, 3-> cancel flight.
    
    def __init__(self, n_apt: int, seed: int = 0) -> None:
        super().__init__()
        
        if seed:
            self.rng = np.random.default_rng(seed=seed)
        else:
            self.rng = np.random.default_rng()
            
        self.n_apt = n_apt
        self.action_space = spaces.MultiDiscrete([n_apt*(n_apt - 1), self.NUM_OF_OPERATIONS])
        
        self.observation_space = spaces.Dict({
            
            "first_hour": spaces.Box(low=-np.inf, high=np.inf, shape=(n_apt*n_apt,), dtype=np.float32),
            "second_hour": spaces.Box(low=-np.inf, high=np.inf, shape=(n_apt*n_apt,), dtype=np.float32),
            "third_hour": spaces.Box(low=-np.inf, high=np.inf, shape=(n_apt*n_apt,), dtype=np.float32)
            
        })
        
    def step(self, action):
        #Parse the action.
        entry_index, operation = action
        #Locate the entry index on the flow matrix.
        #First skip diagonal indices, because they represent self loops on flow matrix.
        diagonal = np.diag_indices(self.n_apt, ndim=2)
        #If entry_index corresponds to a diagonal index, skip to next one.
        diagonal_entries = [self.n_apt*row + column for row, column in zip(diagonal[0], diagonal[1])]
        all_entries = range(self.n_apt*self.n_apt)
        operable_entries = [entry for entry in all_entries if entry not in diagonal_entries]
        
        operation_index = operable_entries[entry_index]
        
        #Operation index.
        if operation == 0:
            pass
        elif operation == 1:
            row, column = divmod(operation_index, self.n_apt)
            if self.hourly_matrix_list[3].iloc[row, column] >= 1:
                self.hourly_matrix_list[3].iloc[row, column] -= 1
            self.hourly_matrix_list[4].iloc[row, column] += 1
            
        elif operation == 2:
            row, column = divmod(operation_index, self.n_apt)
            if self.hourly_matrix_list[3].iloc[row, column] >= 1:
                self.hourly_matrix_list[3].iloc[row, column] -= 1
            self.hourly_matrix_list[5].iloc[row, column] += 1
            
        elif operation == 3:
            row, column = divmod(operation_index, self.n_apt)
            if self.hourly_matrix_list[3].iloc[row, column] >= 1:
                self.hourly_matrix_list[3].iloc[row, column] -= 1
        else:
            raise ValueError("Opeartion action must be one of the (0, 1, 2, 3)")
            
        resulting_observation = self._combine_hourly_flows_and_get_stability_matrices(self.hourly_matrix_list, 
                                                                                     self.recovery_rates)
        reward_from_observation, stability_error = self._reward(operation, resulting_observation, self.past_observation)
        
        if stability_error < 0:
            done = True
        else:
            done = False
            
        self.past_observation = resulting_observation
        
        return resulting_observation, reward_from_observation, done, {}
    
    def reset(self):
        #Get the hourly matrix list. [arr1, arr2, arr3, dep1, dep2, dep3]
        self.hourly_matrix_list = fetch_hourly_flow_matrices(arr_flows, dep_flows, self.rng, random=True)
        
        #Initialize a random recovery rate vector.
        self.recovery_rates = self.rng.normal(loc=1, scale=2, size=(133,))
        
        self.stability_matrices = self._combine_hourly_flows_and_get_stability_matrices(self.hourly_matrix_list, 
                                                                                   self.recovery_rates)
        #Cache the observation for later use.
        self.past_observation = self.stability_matrices
        
        return self.stability_matrices
        
    def _combine_hourly_flows_and_get_stability_matrices(self, hourly_matrix_list, recovery_rates):
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
            
            np.fill_diagonal(matrix, (-1)*recovery_rates)
            matrix = matrix.astype(np.float32)
            stability_matrices.append(matrix)
        
        return {"first_hour": stability_matrices[0].flatten(), 
                "second_hour": stability_matrices[1].flatten(), 
                "third_hour": stability_matrices[2].flatten()}
        
    def _reward(self, operation, observation, past_observation):
        max_real_components = []
        for matrix in observation.values():
            matrix = matrix.reshape(self.n_apt, self.n_apt)
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            max_real_eigenvalue_component = max(eigenvalues.real)
            max_real_components.append(max_real_eigenvalue_component)
            
        max_real_components_past_obs = []
        for matrix in past_observation.values():
            matrix = matrix.reshape(self.n_apt, self.n_apt)
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            max_real_eigenvalue_component = max(eigenvalues.real)
            max_real_components_past_obs.append(max_real_eigenvalue_component)
            
        action_penalty = (operation ** 2) / 10
        eigenvalue_reward = [last_eigenvalue - new_eigenvalue 
                             for last_eigenvalue, new_eigenvalue 
                             in zip(max_real_components_past_obs, max_real_components)]
        
        
        
        return sum(eigenvalue_reward) + action_penalty, sum(eigenvalue_reward)
        
    def close(self):
        pass

env = AirTrafficFlow(n_apt=133)

model = PPO("MultiInputPolicy", env, verbose=1).learn(total_timesteps=1000)




        
       
    