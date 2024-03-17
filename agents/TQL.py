from citylearn.agents.q_learning import TabularQLearning
from typing import List, Mapping
from datetime import datetime
from citylearn.citylearn import CityLearnEnv
from citylearn.reward_function import RewardFunction
from citylearn.wrappers import TabularQLearningWrapper
from ipywidgets import IntProgress

def train_tql(
    agent_kwargs: dict, 
    schema,
    action_bins: List[Mapping[str, int]],
    observation_bins: List[Mapping[str, int]],
    episodes: int, 
    reward_function: RewardFunction,
    random_seed: int
) -> dict:
    """
    Parameters
    ----------
    agent_kwargs: dict
        Defines the hyperparameters used to initialize the SAC agent.
    schema: dict
        Defines the CityLearn environment setup.
    action_bins: List[Mapping[str, int]]
        Defines the number of bins for each active action in each building.
    observation_bins: List[Mapping[str, int]]
        Defines the number of bins for each active observation in each building.
    episodes: int
        Number of episodes to train the agent for.
    reward_function: RewardFunction
        A base or custom reward function class.
    random_seed: int
        Seed for pseudo-random number generator.

    Returns
    -------
    result: dict
        Results from training the agent as well as some input variables
        for reference including the following value keys:

            * random_seed: int
            * env: CityLearnEnv
            * model: DQN
            * actions: List[float]
            * rewards: List[float]
            * agent_kwargs: dict
            * episodes: int
            * reward_function: RewardFunction
            * buildings: List[str]
            * train_start_timestamp: datetime
            * train_end_timestamp: datetime
    """

    # initialize environment
    env = CityLearnEnv(schema, central_agent=True)

    # set reward function
    env.reward_function = reward_function(env=env)

    # set action and observation bin sizes
    observation_bin_sizes = []
    action_bin_sizes = []

    for b in env.buildings:
        observation_bin_sizes.append(observation_bins)
        action_bin_sizes.append(action_bins)

    # discretize action space
    env = TabularQLearningWrapper(
        env.unwrapped,
        observation_bin_sizes=observation_bin_sizes,
        action_bin_sizes=action_bin_sizes
    )

    total_timesteps = episodes*(env.time_steps - 1)

    m = env.observation_space[0].n
    n = env.action_space[0].n
    t = env.time_steps - 1
    tql_episodes = m*n*episodes/t
    tql_episodes = int(tql_episodes)
    print('Number of episodes to train:', episodes)

    loader = get_loader(max=tql_episodes*t)
    display(loader)

    model = CustomTabularQLearning(
        env=env,
        loader=loader,
        random_seed=random_seed,
        **agent_kwargs
    )

    # train agent
    train_start_timestamp = datetime.now()
    model = model.learn(episodes=tql_episodes)
    train_end_timestamp = datetime.now()

    # evaluate agent
    observations = env.reset()
    actions_list = []

    while not env.done:
        actions, _ = model.predict(observations, deterministic=True)
        observations, _, _, _ = env.step(actions)
        actions_list.append(actions)

    kpis = env.evaluate()
    kpis = kpis.pivot(index='cost_function', columns='name', values='value')
    kpis = kpis.dropna(how='all')
    display(kpis)


    return {
        'random_seed': random_seed,
        'env': env,
        'model': model,
        'actions': actions_list,
        # 'rewards': rewards,
        'agent_kwargs': agent_kwargs,
        'episodes': episodes,
        'reward_function': reward_function,
        'train_start_timestamp': train_start_timestamp,
        'train_end_timestamp': train_end_timestamp,
    }

class CustomTabularQLearning(TabularQLearning):
    def __init__(
        self, env: CityLearnEnv, loader: IntProgress,
        random_seed: int = None, **kwargs
    ):
        r"""Initialize CustomRBC.

        Parameters
        ----------
        env: Mapping[str, CityLearnEnv]
            CityLearn environment instance.
        loader: IntProgress
            Progress bar.
        random_seed: int
            Random number generator reprocucibility seed for
            eqsilon-greedy action selection.
        kwargs: dict
            Parent class hyperparameters
        """

        super().__init__(env=env, random_seed=random_seed, **kwargs)
        self.loader = loader
        self.reward_history = []

    def next_time_step(self):
        if self.env.time_step == 0:
            self.reward_history.append(0)

        else:
            self.reward_history[-1] += sum(self.env.rewards[-1])

        self.loader.value += 1
        super().next_time_step()


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