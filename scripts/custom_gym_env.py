import gym
from gym import spaces
import numpy as np
import pandas as pd

from utils import fetch_single_flow_matrix

flight_flows = pd.read_pickle("../data/all_time_flight_flow_df.pickle")

class AirTrafficFlow(gym.Env):
    
    MAX_FLIGHT_FLOW_ENTRY = 20 #Discrete limit is chosen as 20 because maximum observed entry on flight flows is 17.
    NUM_OF_OPERATIONS = 4 #0-> do nothing, 1-> delay 1hr, 2-> delay 2h, 3-> cancel flight.
    
    def __init__(self, n_apt: int) -> None:
        super().__init__()
        
        self.action_space = spaces.MultiDiscrete([n_apt*(n_apt - 1), NUM_OF_OPERATIONS])
        
        self.observation_space = spaces.MultiDiscrete([MAX_FLIGHT_FLOW_ENTRY for _ in range(n_apt*(n_apt - 1))])
        
        
    def step(self, action):
        #Parse the action.
        entry_index, operation = action
        #Locate the entry index on the flow matrix.
        #First skip diagonal indices, because they represent self loops on flow matrix.
        diagonal = np.diag_indices(n_apt, ndim=2)
        #If entry_index corresponds to a diagonal index, skip to next one.
        if entry_index in diagonal:
            entry_index += 1
            
    def reset(self):
        self.flight_flow_matrix = fetch_single_flow_matrix(flight_flows, random=True)
        
       
    