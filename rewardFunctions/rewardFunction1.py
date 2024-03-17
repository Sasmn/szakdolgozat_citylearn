from citylearn.reward_function import RewardFunction
from citylearn.citylearn import CityLearnEnv
import numpy as np
from typing import List, Mapping, Union

class CustomRewardFunction(RewardFunction):
    def __init__(self, env: CityLearnEnv):
        super().__init__(env)
        self.env=env

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        
        aggregated_consumption = sum(b.net_electricity_consumption[-1] for b in self.env.buildings)
        aggregated_consumption_without_storage = sum(b.net_electricity_consumption_without_storage[-1] for b in self.env.buildings)
        aggregated_storage_consumption = aggregated_consumption_without_storage - aggregated_consumption

        peak_consumption_penalty = 0
        if aggregated_storage_consumption > 0:
            peak_consumption_penalty = 1 / (1 + aggregated_consumption_without_storage) * 1 / ( 1 + aggregated_storage_consumption)
        else:
            peak_consumption_penalty = np.power(aggregated_consumption_without_storage, 3) * (1 + abs(aggregated_storage_consumption))

        reward = [-peak_consumption_penalty]

        return reward