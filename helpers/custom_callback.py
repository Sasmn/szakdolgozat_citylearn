from stable_baselines3.common.callbacks import BaseCallback
from citylearn.citylearn import CityLearnEnv

class CustomCallback(BaseCallback):
    def __init__(self, env: CityLearnEnv):

        super().__init__(verbose=0)
        self.env = env
        self.reward_history = [0]
        self.learning_rates = []


        self.current_timestep = 0
    def _on_step(self) -> bool:
        if self.env.time_step == 0:
            self.reward_history.append(0)

        else:
            self.reward_history[-1] += sum(self.env.rewards[-1])

        self.current_timestep += 1
        return True