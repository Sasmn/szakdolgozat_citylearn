from ipywidgets import IntProgress
from citylearn.agents.rbc import HourRBC
from citylearn.citylearn import CityLearnEnv
from typing import Mapping

from datetime import datetime
from citylearn.citylearn import CityLearnEnv
from citylearn.reward_function import RewardFunction
from ipywidgets import IntProgress

def train_rbc(
    action_map: dict, 
    schema,
    episodes: int, 
    reward_function: RewardFunction,
) -> dict:
    """

    Parameters
    ----------
    action_map: dict
        Defines the action map used by the agent.
    schema: dict
        Defines the CityLearn environment setup.
    episodes: int
        Number of episodes to train the agent for.

    Returns
    -------
    result: dict
        Results from training the agent as well as some input variables
        for reference including the following value keys:

            * env: CityLearnEnv
            * model: RBC
            * rewards: List[float]
            * action_map: dict
            * episodes: int
            * buildings: List[str]
            * train_start_timestamp: datetime
            * train_end_timestamp: datetime
    """

    # initialize environment
    env = CityLearnEnv(schema, central_agent=True)

    # set reward function
    env.reward_function = reward_function(env=env)

    loader = get_loader(max=episodes*(schema['simulation_end_time_step']-1))
    display(loader)

    # initialize agent
    model = CustomRBC(env, action_map, loader)

    total_timesteps = episodes*(env.time_steps - 1)
    print('Number of episodes to train:', episodes)
    

    # train agent
    train_start_timestamp = datetime.now()
    model = model.learn(episodes)
    train_end_timestamp = datetime.now()

    kpis = env.evaluate()
    kpis = kpis.pivot(index='cost_function', columns='name', values='value')
    kpis = kpis.dropna(how='all')
    display(kpis)


    return {
        'env': env,
        'model': model,
        # 'rewards': rewards,
        'action_map': action_map,
        'episodes': episodes,
        'train_start_timestamp': train_start_timestamp,
        'train_end_timestamp': train_end_timestamp,
    }



class CustomRBC(HourRBC):
    def __init__(
        self, env: CityLearnEnv, 
        action_map: Mapping[int, float] = None,
        loader: IntProgress = None
    ):

        super().__init__(env=env, action_map=action_map)
        self.loader = loader

    def next_time_step(self):
        super().next_time_step()

        if self.loader is not None:
            self.loader.value += 1
        else:
            pass

def get_loader(**kwargs):
    """Returns a progress bar"""

    kwargs = {
        'value': 0,
        'min': 0,
        'max': 10,
        'description': 'Simulating:',
        'bar_style': '',
        'style': {'bar_color': 'maroon'},
        'orientation': 'horizontal',
        **kwargs
    }
    return IntProgress(**kwargs)