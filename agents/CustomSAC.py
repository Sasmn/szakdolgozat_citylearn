from ipywidgets import IntProgress
from citylearn.agents.rlc import RLC
from citylearn.citylearn import CityLearnEnv
from typing import Mapping

class CustomSAC(RLC):
   def __init__(
       self, env: CityLearnEnv,
       loader: IntProgress = None
    ):
      r"""Initialize CustomRBC.

      Parameters
      ----------
      env: Mapping[str, CityLearnEnv]
         CityLearn environment instance.
      action_map: Mapping[int, float]
         Mapping of hour to control action.
      loader: IntProgress
         Progress bar.
      """

      super().__init__(env=env)
      self.loader = loader

   def next_time_step(self):
      r"""Advance to next `time_step`."""

      super().next_time_step()

      if self.loader is not None:
         self.loader.value += 1
      else:
         pass