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
        reward = [
            (-np.sqrt(abs(self.env.net_electricity_consumption[-1] - self.env.net_electricity_consumption[-2]))) * abs(self.env.net_electricity_consumption[-1] - np.mean(self.env.net_electricity_consumption))
        ]

        return reward