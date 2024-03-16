from gym import spaces, ActionWrapper
from citylearn.citylearn import CityLearnEnv
from typing import List, Mapping
from wrappers.DiscreteActionWrapper import DiscreteActionWrapper
import itertools
import numpy as np
import sys

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
        self.options = self.get_options()
        self.action_lengths = self.get_action_lenghts()
        self.combinations_lenght = self.get_combinations_length()

    @property
    def action_space(self) -> spaces.Discrete:
        """Returns action space for discretized actions."""
        return spaces.Discrete(self.combinations_lenght)
    
    def action(self, action: float) -> List[int]:
        """Returns discretized action."""
        return self.get_combination(int(action))
    
    def get_combination(self, action: int) -> List[int]:
        """Returns a discretized action combination given an action."""

        lengths = self.action_lengths
        total_combinations = self.combinations_lenght

        combination = []

        for length in reversed(lengths):
            total_combinations //= length
            quotient, action = divmod(action, total_combinations)
            combination.append(self.options[len(lengths) - len(combination) - 1][quotient])
    
        return combination
    
    def get_combinations_length(self) -> int:
        """Returns the length of all combinations of discrete actions."""

        lengths = [len(sublist) for sublist in self.options]
        total_combinations = 1
        for length in lengths:
            total_combinations *= length
        return total_combinations
    
    def get_options(self) -> List[List[int]]:
        """Returns all possible set of discrete values for the actions."""

        options = [list(np.arange(d.n, dtype=np.int8)) for d in self.env.action_space[0]]
        return options
    
    def get_action_lenghts(self) -> List[int]:
        """Returns the length of all combinations of discrete actions."""

        lengths = [len(sublist) for sublist in self.options]
        return lengths