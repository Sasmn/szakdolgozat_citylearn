from stable_baselines3.common.callbacks import BaseCallback
from citylearn.citylearn import CityLearnEnv

class CustomCallback(BaseCallback):
    def __init__(self, env: CityLearnEnv, total_timesteps: int):

        super().__init__(verbose=0)
        self.env = env
        self.reward_history = [0]
        self.learning_rates = []
        self.total_timesteps = total_timesteps


        self.current_timestep = 0
    def _on_step(self) -> bool:
        if self.env.time_step == 0:
            self.reward_history.append(0)

        else:
            self.reward_history[-1] += sum(self.env.rewards[-1])

        self.current_timestep += 1
        # self.learning_rates.append(self.model.learning_rate(self.total_timesteps - self.current_timestep))
        return True