from citylearn.reward_function import RewardFunction
from citylearn.citylearn import CityLearnEnv
import numpy as np
from typing import List, Mapping, Union

class CustomRewardFunction(RewardFunction):
    def __init__(self, env: CityLearnEnv):
        super().__init__(env)
        self.env=env

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        reward_list = []

        for b in self.env.buildings:
            # cost = b.net_electricity_consumption_cost[-1]
            # battery_capacity = b.electrical_storage.capacity_history[0]
            # battery_soc = b.electrical_storage.soc[-1]/battery_capacity
            # net_consumption = b.net_electricity_consumption[-1]
            # # print("battery_soc: ", b.electrical_storage.soc[-1])
            # # print("battery_capacity: ",battery_capacity)
            # # print("battery: ",battery_soc)
            # # print("\n")
            # battery_charging = b.electrical_storage_electricity_consumption[-1]
            # # penalty1 = -(1.0 + np.sign(net_consumption) * battery_soc)
            # # penalty2 = -(1.0 + np.sign(b.solar_generation[-1] + b.net_electricity_consumption_without_storage[-1]) * battery_charging / battery_capacity)
            # penalty2 = ((b.solar_generation[-1] + 0.5) * battery_soc * net_consumption / max(b.net_electricity_consumption))
            # ratio_of_grid_consumption = b.net_electricity_consumption_without_storage_and_pv[-1] / (b.net_electricity_consumption_without_storage_and_pv[-1] + abs(b.solar_generation[-1]) - b.electrical_storage_electricity_consumption[-1])
            # penalty1 = b.solar_generation[-1] * -5 * b.electrical_storage_electricity_consumption[-1]
            # penalty2 = ratio_of_grid_consumption

            
            
            
            # reward = penalty * ( 1 / (battery_soc + 0.001) ) * abs(cost)
            # ha 2 alatt van, akkor inkább töltsük a batteryt, és minnél inkább fölötte, annál inkább pótoljuk ki a consumption-t a battery-vel.
            # reward = np.power(penalty1, 1) - np.power(((b.net_electricity_consumption_without_storage_and_pv[-1] + b.electrical_storage_electricity_consumption[-1])), 2)
            # reward = -1 * (b.net_electricity_consumption[-1] +  1 / (1 + b.electrical_storage_electricity_consumption[-1]) + 1 / (1 + b.solar_generation[-1]))


            # consumption_change_penalty = np.power((b.net_electricity_consumption[-1] - b.net_electricity_consumption[-2]), 2)
            # alpha = 2
            # beta = 0.5
            # peak_consumption_peanlty = alpha * np.exp(beta * b.net_electricity_consumption[-1])

            # reward = -consumption_change_penalty - peak_consumption_peanlty


            # window_size = 4
            # moving_avg = np.convolve(b.net_electricity_consumption, np.ones(window_size)/window_size, mode='valid')
            # deviation = np.abs(b.net_electricity_consumption[window_size-1:] - moving_avg)
            # threshold_multiplier = 2
            # threshold = np.mean(deviation) + threshold_multiplier * np.std(deviation)

            # if deviation > threshold:
            #     peak_consumption_peanlty = deviation - threshold
            # else:
            #     0
                
            # reward = -consumption_change_penalty - peak_consumption_peanlty
                


            delta_consumption = b.net_electricity_consumption[-1] - b.net_electricity_consumption[-2]
            consumption_change_penalty = 0
            if delta_consumption > 0:
                consumption_change_penalty = max(np.power(delta_consumption, 2), b.net_electricity_consumption[-1])
            else:
                consumption_change_penalty = 0
            
            reward = -consumption_change_penalty

            
            # reward = - 1 * max(b.net_electricity_consumption[-24:])

            reward_list.append(reward)

        reward = [sum(reward_list)]

        return reward