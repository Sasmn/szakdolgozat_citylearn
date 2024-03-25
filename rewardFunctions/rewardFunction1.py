from citylearn.reward_function import RewardFunction
from citylearn.citylearn import CityLearnEnv
import numpy as np
from typing import List, Mapping, Union
import math

class CustomRewardFunction(RewardFunction):
    def __init__(self, env: CityLearnEnv):
        super().__init__(env)
        self.env=env

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        
        # aggregated_consumption = sum(b.net_electricity_consumption[-1] for b in self.env.buildings)
        # aggregated_consumption_without_storage = sum(b.net_electricity_consumption_without_storage[-1] for b in self.env.buildings)
        # aggregated_storage_consumption = aggregated_consumption_without_storage - aggregated_consumption

        # peak_consumption_penalty = 0
        # if aggregated_storage_consumption > 0:
        #     peak_consumption_penalty = 1 / (1 + aggregated_consumption_without_storage) * 1 / ( 1 + aggregated_storage_consumption)
        # else:
        #     peak_consumption_penalty = np.power(aggregated_consumption_without_storage, 3) * (1 + abs(aggregated_storage_consumption))

        # reward = [-peak_consumption_penalty]

        # aggregated_consumption = sum(b.net_electricity_consumption[-1] for b in self.env.buildings)
        # aggregated_consumption_without_storage = sum(b.net_electricity_consumption_without_storage[-1] for b in self.env.buildings)
        # aggregated_storage_consumption = aggregated_consumption_without_storage - aggregated_consumption

        # peak_consumption_penalty = 0
        # if aggregated_storage_consumption > 0:
        #     peak_consumption_penalty = 1 / (1 + aggregated_consumption_without_storage) * 1 / ( 1 + aggregated_storage_consumption)
        # else:
        #     peak_consumption_penalty = np.power(aggregated_consumption_without_storage, 2) * (1 + abs(aggregated_storage_consumption))

        # reward = [-peak_consumption_penalty]


        aggregated_consumption = sum(b.net_electricity_consumption[-1] for b in self.env.buildings)
        aggregated_consumption_without_storage = sum(b.net_electricity_consumption_without_storage[-1] for b in self.env.buildings)
        aggregated_storage_consumption = aggregated_consumption_without_storage - aggregated_consumption

        arrays = [b.net_electricity_consumption_without_storage for b in self.env.buildings]

        # Sum the arrays element-wise
        sum_array = np.sum(arrays, axis=0)
        avg_consumption_without_storage = np.mean(sum_array)

        peak_consumption_penalty = 0
        if aggregated_storage_consumption > 0:
            # peak_consumption_penalty = aggregated_consumption_without_storage * 1 / ( 1 + aggregated_storage_consumption)
            # peak_consumption_penalty = math.exp((avg_consumption_without_storage - aggregated_consumption_without_storage))
            # peak_consumption_penalty = math.exp((avg_consumption_without_storage - aggregated_consumption_without_storage)) + aggregated_consumption_without_storage * 1 / (1 + aggregated_storage_consumption)
            peak_consumption_penalty = 1 / (1 + math.exp((avg_consumption_without_storage - aggregated_consumption_without_storage))) * aggregated_consumption_without_storage * 1 / (1 + aggregated_storage_consumption)
            
        else:
            peak_consumption_penalty = np.power(aggregated_consumption_without_storage, 2) * (1 + abs(aggregated_storage_consumption))

        reward = [-peak_consumption_penalty]


        # aggregated_consumption = sum(b.net_electricity_consumption[-1] for b in self.env.buildings)
        # aggregated_consumption_without_storage = sum(b.net_electricity_consumption_without_storage[-1] for b in self.env.buildings)
        # aggregated_storage_consumption = aggregated_consumption_without_storage - aggregated_consumption

        # arrays = [b.net_electricity_consumption_without_storage for b in self.env.buildings]
        # sum_array = np.sum(arrays, axis=0)
        # avg_consumption_without_storage = np.mean(sum_array)

        # arrays = [b.net_electricity_consumption for b in self.env.buildings]
        # sum_array = np.sum(arrays, axis=0)
        # avg_consumption = np.mean(sum_array)

        # peak_consumption_penalty = math.exp(2 * (aggregated_consumption - avg_consumption))

        # reward = [-peak_consumption_penalty]

        arrays = [b.net_electricity_consumption_without_storage for b in self.env.buildings]
        sum_array = np.sum(arrays, axis=0)
        previous_max_delta = None
    
        for i in range(1, len(sum_array)):
            delta = abs(sum_array[i] - avg_consumption_without_storage)
            if previous_max_delta is None or delta > previous_max_delta:
                previous_max_delta = delta

        arrays2 = [b.electrical_storage.capacity for b in self.env.buildings]
        sum_array2 = np.sum(arrays2, axis=0)

        storage_consumptions = sum_array - sum_array2

        max_storage_consumption = None
        for i in range(1, len(storage_consumptions)):
            if max_storage_consumption is None or storage_consumptions[i] > max_storage_consumption:
                max_storage_consumption = storage_consumptions[i]


        reward = [-np.nan_to_num(math.exp(np.nan_to_num((aggregated_consumption_without_storage - avg_consumption_without_storage) / previous_max_delta)) * aggregated_storage_consumption / max_storage_consumption, 0.0)]

        return reward