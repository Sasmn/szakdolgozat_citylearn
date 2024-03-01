from gym import spaces, ActionWrapper
from citylearn.citylearn import CityLearnEnv
from typing import List, Mapping
from wrappers.DiscreteActionWrapper import DiscreteActionWrapper
import itertools

class DQNActionWrapper(ActionWrapper):
    """Action wrapper for :py:class:`citylearn.agents.q_learning.TabularQLearning` agent.

    Wraps `env` in :py:class:`citylearn.wrappers.DiscreteActionWrapper`.
    
    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment.
    bin_sizes: List[Mapping[str, int]], optional
        Then number of bins for each active action in each building.
    default_bin_size: int, default = 10
        The default number of bins if `bin_sizes` is unspecified for any active building action.
    """

    def __init__(self, env: CityLearnEnv, bin_sizes: List[Mapping[str, int]] = None, default_bin_size: int = None) -> None:
        env = DiscreteActionWrapper(env, bin_sizes=bin_sizes, default_bin_size=default_bin_size)
        super().__init__(env)
        self.env: CityLearnEnv
        self.combinations = self.set_combinations()

    @property
    def action_space(self) -> spaces.Discrete:
        """Returns action space for discretized actions."""
        return spaces.Discrete(len(self.combinations))
    
    def action(self, action: float) -> List[int]:
        """Returns discretized action."""
        return self.combinations[int(action)]
    
    def set_combinations(self) -> List[int]:
        """Returns all combinations of discrete actions."""
        options = [list(range(d.n)) for d in self.env.action_space[0]]
        combs = [comb for comb in itertools.product(*options)]
        return combs