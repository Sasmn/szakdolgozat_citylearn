from ipywidgets import IntProgress
from citylearn.agents.rbc import HourRBC
from citylearn.citylearn import CityLearnEnv
from typing import Mapping

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