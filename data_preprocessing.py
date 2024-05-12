import numpy as np
from citylearn.utilities import read_json
import os

def getData(
    active_actions,
    number_of_buildings,
    number_of_days,
    root_path
):

    # read the schema
    dataset_name = 'citylearn_challenge_2023_phase_3_3'
    root_directory = os.path.join(f"{root_path}data", dataset_name)
    filepath = os.path.join(root_directory, 'schema.json')
    schema = read_json(filepath)
    schema['root_directory'] = root_directory

    random_seed = 42
    np.random.seed(random_seed)

    buildings = list(schema['buildings'].keys())
    buildings = np.random.choice(buildings, size=number_of_buildings, replace=False).tolist()

    print("Active buildings in the schema: ")
    for building in schema['buildings']:
        if building in buildings:
            schema['buildings'][building]['include'] = True
            print(building)
        else:
            schema['buildings'][building]['include'] = False


    print("\n\nActive observations in the schema: ")
    active_observations = [
        'day_type',
        'hour', 
        'outdoor_dry_bulb_temperature',
        'outdoor_dry_bulb_temperature_predicted_6h',
        'outdoor_dry_bulb_temperature_predicted_12h',
        'outdoor_dry_bulb_temperature_predicted_24h',
        'outdoor_relative_humidity',
        'outdoor_relative_humidity_predicted_6h',
        'outdoor_relative_humidity_predicted_12h',
        'outdoor_relative_humidity_predicted_24h',
        'diffuse_solar_irradiance', 
        'diffuse_solar_irradiance_predicted_6h',
        'diffuse_solar_irradiance_predicted_12h',
        'diffuse_solar_irradiance_predicted_24h',
        'direct_solar_irradiance',
        'direct_solar_irradiance_predicted_6h',
        'direct_solar_irradiance_predicted_12h',
        'direct_solar_irradiance_predicted_24h',
        'carbon_intensity',
        'indoor_dry_bulb_temperature', 
        'indoor_relative_humidity',
        'non_shiftable_load', 
        'solar_generation', 
        'dhw_storage_soc',
        'electrical_storage_soc'
        'net_electricity_consumption',
        'electricity_pricing', 
        'cooling_device_cop', 
        'cooling_demand', 
        'dhw_demand', 
        'cooling_electricity_consumption', 
        'dhw_electricity_consumption', 
        'occupant_count'
        'indoor_dry_bulb_temperature_set_point', 
        'indoor_dry_bulb_temperature_delta'
    ]
    for observation in schema['observations']:
        if active_observations == 'no_change':
            if schema['observations'][observation]['active'] == True:
                print(observation)
        elif active_observations == 'all':
            schema['observations'][observation]['active'] = True
            schema['observations'][observation]['shared_in_central_agent'] = True
            print(observation)
        else:
            if observation in active_observations:
                schema['observations'][observation]['active'] = True
                schema['observations'][observation]['shared_in_central_agent'] = True
                print(observation)
            else:
                schema['observations'][observation]['active'] = False
                schema['observations'][observation]['shared_in_central_agent'] = False

    print("\n\nActive actions in the schema: ")
    number_of_actions = len(active_actions)
    for action in schema['actions']:
        if action in active_actions:
            schema['actions'][action]['active'] = True
            print(action)
        else:
            schema['actions'][action]['active'] = False

    # Set simulation timeframe (episodes)
    schema['simulation_start_time_step'] = 1
    lenght_of_simulation_in_days = number_of_days
    schema['simulation_end_time_step'] = lenght_of_simulation_in_days * 24 + schema['simulation_start_time_step'] - 1
    print(f"\n\nSimulation timeframe: {schema['simulation_start_time_step']} to {schema['simulation_end_time_step']}")

    for building in schema['buildings']:
        # Set custom pricing helper buildig
        schema['buildings'][building]['type'] = 'wrappers.CustomPricingWrapper.CustomPricingBuilding'

        # Set pricing data
        schema['buildings'][building]['pricing'] = 'pricing_data_mvm_A2.csv'

        # Set carbon intensity data
        schema['buildings'][building]['carbon_intensity'] = 'carbon_intensity_HU.csv'

        # Set weather data
        schema['buildings'][building]['weather'] = 'weather_HU.csv'

    return (
        schema,
        number_of_buildings,
        number_of_actions,
        random_seed
    )