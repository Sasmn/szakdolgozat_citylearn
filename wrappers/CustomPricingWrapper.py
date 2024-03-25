import numpy as np
from citylearn.building import LSTMDynamicsBuilding
from citylearn.energy_model import HeatPump
from citylearn.data import Pricing
from citylearn.wrappers import Wrapper

class CustomPricingBuilding(LSTMDynamicsBuilding):
    def update_variables(self):
        """Update cooling, heating, dhw and net electricity consumption as well as net electricity consumption cost and carbon emissions."""

        if self.time_step == 0:
            temperature = self.weather.outdoor_dry_bulb_temperature[self.time_step]

            # cooling electricity consumption
            cooling_demand = self._Building__energy_from_cooling_device[self.time_step] + self.cooling_storage.energy_balance[self.time_step]
            cooling_electricity_consumption = self.cooling_device.get_input_power(cooling_demand, temperature, heating=False)
            self.cooling_device.update_electricity_consumption(cooling_electricity_consumption)

            # heating electricity consumption
            heating_demand = self._Building__energy_from_heating_device[self.time_step] + self.heating_storage.energy_balance[self.time_step]

            if isinstance(self.heating_device, HeatPump):
                heating_electricity_consumption = self.heating_device.get_input_power(heating_demand, temperature, heating=True)
            else:
                heating_electricity_consumption = self.dhw_device.get_input_power(heating_demand)

            self.heating_device.update_electricity_consumption(heating_electricity_consumption)

            # dhw electricity consumption
            dhw_demand = self._Building__energy_from_dhw_device[self.time_step] + self.dhw_storage.energy_balance[self.time_step]

            if isinstance(self.dhw_device, HeatPump):
                dhw_electricity_consumption = self.dhw_device.get_input_power(dhw_demand, temperature, heating=True)
            else:
                dhw_electricity_consumption = self.dhw_device.get_input_power(dhw_demand)

            self.dhw_device.update_electricity_consumption(dhw_electricity_consumption)

            # non shiftable load electricity consumption
            non_shiftable_load_electricity_consumption = self._Building__energy_to_non_shiftable_load[self.time_step]
            self.non_shiftable_load_device.update_electricity_consumption(non_shiftable_load_electricity_consumption)

            # electrical storage
            electrical_storage_electricity_consumption = self.electrical_storage.energy_balance[self.time_step]
            self.electrical_storage.update_electricity_consumption(electrical_storage_electricity_consumption, enforce_polarity=False)

        else:
            pass

        # net electricity consumption
        net_electricity_consumption = 0.0

        if not self.power_outage:
            net_electricity_consumption = self.cooling_device.electricity_consumption[self.time_step] \
                + self.heating_device.electricity_consumption[self.time_step] \
                    + self.dhw_device.electricity_consumption[self.time_step] \
                        + self.non_shiftable_load_device.electricity_consumption[self.time_step] \
                            + self.electrical_storage.electricity_consumption[self.time_step] \
                                + self.solar_generation[self.time_step]
        else:
            pass

        self._Building__net_electricity_consumption[self.time_step] = net_electricity_consumption

        consumption_until_now = np.sum(self._Building__net_electricity_consumption[:self.time_step + 1])
        pricing = self.pricing.electricity_pricing[self.time_step]

        if consumption_until_now > 210.25:
            pricing = 0.194733
        elif net_electricity_consumption < 0:
            pricing = 0

        self.pricing.electricity_pricing[self.time_step] = pricing
        # net electriciy consumption cost
        self._Building__net_electricity_consumption_cost[self.time_step] = net_electricity_consumption * pricing

        # net electriciy consumption emission
        self._Building__net_electricity_consumption_emission[self.time_step] = max(0.0, net_electricity_consumption*self.carbon_intensity.carbon_intensity[self.time_step])


    @property
    def pricing(self) -> Pricing:
        """Energy pricing and forecasts time series."""

        return self._Building__pricing
    
    @pricing.setter
    def pricing(self, pricing: Pricing):
        if pricing is None:
            self._Building__pricing = Pricing(
                np.zeros(self.episode_tracker.simulation_time_steps + 1, dtype='float32'),
                np.zeros(self.episode_tracker.simulation_time_steps + 1, dtype='float32'),
                np.zeros(self.episode_tracker.simulation_time_steps + 1, dtype='float32'),
                np.zeros(self.episode_tracker.simulation_time_steps + 1, dtype='float32'),
            )
        else:
            self._Building__pricing = pricing