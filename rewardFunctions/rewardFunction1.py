from citylearn.reward_function import RewardFunction
from citylearn.citylearn import CityLearnEnv
import numpy as np
from typing import List, Mapping, Union

class CustomRewardFunction(RewardFunction):
    def __init__(self, env: CityLearnEnv):
        super().__init__(env)
        self.env=env

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        # print(observations)

        reward_list = []
        for b in self.env.buildings:       
            peak_importance = 0.8
            peak_consumption = max(b.net_electricity_consumption)
            current_consumption = b.net_electricity_consumption[-1]
            penalty1_temperature = abs(b.solar_generation[-1]) + 0.001

            penalty1 = 1 / (abs(peak_consumption - current_consumption) + penalty1_temperature)
            print("peak: ", peak_consumption)
            print("current: ", current_consumption)
            print("solar: ", abs(b.solar_generation[-1]) + 0.001)
            print("penal1: ", penalty1)
            print("\n")

            cost = b.net_electricity_consumption_cost[-1]

            battery_capacity = b.electrical_storage.capacity_history[0]
            battery_soc = b.electrical_storage.soc[-1]/battery_capacity

            penalty2 = -(1.0 + np.sign(cost)*battery_soc)
            # print(penalty2)
            
            reward = - peak_importance * penalty1 - (1 - peak_importance) * penalty2 * abs(cost)

            reward_list.append(reward)

        reward = [sum(reward_list)]
        # print(reward)

        return reward