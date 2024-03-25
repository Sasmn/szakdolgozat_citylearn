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
        district_consumption = sum(b.net_electricity_consumption[-1] for b in self.env.buildings)

        building_consumptions = [b.net_electricity_consumption for b in self.env.buildings]
        district_consumptions = np.sum(building_consumptions, axis=0)
        avg_district_consumption = np.mean(district_consumptions)

        building_consumptions_without_storage = [b.net_electricity_consumption_without_storage for b in self.env.buildings]
        district_consumptions_without_storage = np.sum(building_consumptions_without_storage, axis=0)
        avg_district_consumption_without_storage = np.mean(district_consumptions_without_storage)

        previous_max_deviation = None
        for i in range(1, len(district_consumptions)):
            delta = abs(district_consumptions[i] - avg_district_consumption_without_storage)
            if previous_max_deviation is None or delta > previous_max_deviation:
                previous_max_deviation = delta

        deviaton_penalty = 0
        if (previous_max_deviation > 0):
            deviaton_penalty = (district_consumption - avg_district_consumption) / previous_max_deviation

        reward = [
            (math.cos(deviaton_penalty * math.pi) - 1) * np.power(abs((district_consumption - avg_district_consumption)), 3)
        ]

        return reward