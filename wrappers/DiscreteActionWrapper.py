import numpy as np
from gym import spaces, ActionWrapper
from citylearn.citylearn import CityLearnEnv
from typing import List, Mapping

class DiscreteActionWrapper(ActionWrapper):
    """Wrapper for action space discretization.

    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment.
    bin_sizes: List[Mapping[str, int]], optional
        Then number of bins for each active action in each building.
    default_bin_size: int, default = 10
        The default number of bins if `bin_sizes` is unspecified for any active building action.
    """

    def __init__(self, env: CityLearnEnv, bin_sizes: List[Mapping[str, int]] = None, default_bin_size: int = None):
        super().__init__(env)
        self.env: CityLearnEnv
        assert bin_sizes is None or len(bin_sizes) == len(self.env.buildings), 'length of bin_size must equal number of buildings.'
        self.bin_sizes = [{} for _ in self.env.buildings] if bin_sizes is None else bin_sizes
        self.default_bin_size = 10 if default_bin_size is None else default_bin_size
        self.bin_sizes = [
            {a: s.get(a, self.default_bin_size) for a in b.active_actions} 
            for b, s in zip(self.env.buildings, self.bin_sizes)
        ]
        
    @property
    def action_space(self) -> List[spaces.MultiDiscrete]:
        """Returns action space for discretized actions."""

        if self.env.central_agent:
            bin_sizes = []

            for b in self.bin_sizes:
                for _, v in b.items():
                    bin_sizes.append(v)
            
            action_space = [spaces.MultiDiscrete(bin_sizes)]

        else:
            action_space = [spaces.MultiDiscrete(list(b.values())) for b in self.bin_sizes]

        return action_space

    def action(self, actions: List[float]) -> List[float]:
        """Returns undiscretized actions."""

        transformed_actions = []
        consinuous_action_space = self.env.unwrapped.action_space[0]
        discrete_action_space = self.action_space[0]
        
        transformed_actions = []
        
        for j, (lower_limit, upper_limit, bin_size) in enumerate(zip(consinuous_action_space.low, consinuous_action_space.high, discrete_action_space)):
            a = np.linspace(lower_limit, upper_limit, bin_size.n)[actions[j]]
            transformed_actions.append(a)
    
        return transformed_actions