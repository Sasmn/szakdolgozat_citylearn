from typing import List, Mapping
from datetime import datetime
from citylearn.citylearn import CityLearnEnv
from citylearn.reward_function import RewardFunction
from stable_baselines3.a2c import A2C
from helpers.custom_callback import CustomCallback
from citylearn.wrappers import NormalizedObservationWrapper
from citylearn.wrappers import StableBaselines3Wrapper

def train_a2c(
    agent_kwargs: dict, 
    schema,
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
            * model: A2C
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

    # wrap environment
    env = NormalizedObservationWrapper(env)
    env = StableBaselines3Wrapper(env)

    # initialize agent
    model = A2C('MlpPolicy', env, **agent_kwargs, seed=random_seed)

    total_timesteps = episodes*(env.time_steps - 1)
    print('Number of episodes to train:', episodes)
    
    # initialize callback
    callback = CustomCallback(env=env, total_timesteps=total_timesteps)

    # train agent
    train_start_timestamp = datetime.now()
    model = model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
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

    # get rewards
    rewards = callback.reward_history[:episodes]


    return {
        'random_seed': random_seed,
        'env': env,
        'model': model,
        'actions': actions_list,
        'rewards': rewards,
        'agent_kwargs': agent_kwargs,
        'episodes': episodes,
        'reward_function': reward_function,
        'train_start_timestamp': train_start_timestamp,
        'train_end_timestamp': train_end_timestamp,
    }