import pandas as pd
from IPython.display import display

def print_env_data(env):
    # electrical storage
    print('Electrical storage capacity:', {
        b.name: b.electrical_storage.capacity for b in env.buildings
    })
    print('Electrical storage nominal power:', {
        b.name: b.electrical_storage.nominal_power for b in env.buildings
    })
    print('Electrical storage capacity history:', {
        b.name: b.electrical_storage.capacity_history for b in env.buildings
    })
    print('Electrical storage loss_coefficient:', {
        b.name: b.electrical_storage.loss_coefficient for b in env.buildings
    })
    print('Electrical storage initial_soc:', {
        b.name: b.electrical_storage.initial_soc for b in env.buildings
    })
    print('Electrical storage soc:', {
        b.name: b.electrical_storage.soc for b in env.buildings
    })
    print('Electrical storage efficiency:', {
        b.name: b.electrical_storage.efficiency for b in env.buildings
    })
    print('Electrical storage efficiency history:', {
        b.name: b.electrical_storage.efficiency_history for b in env.buildings
    })
    print('Electrical storage electricity consumption:', {
        b.name: b.electrical_storage.electricity_consumption
        for b in env.buildings
    })
    print('Electrical storage capacity loss coefficient:', {
        b.name: b.electrical_storage.loss_coefficient for b in env.buildings
    })
    print()
    # pv
    print('PV nominal power:', {
        b.name: b.pv.nominal_power for b in env.buildings
    })
    print()

    # active observations and actions
    with pd.option_context(
        'display.max_rows', None,
        'display.max_columns', None,
        'display.width', None
    ):
        print('Active observations:')
        display(pd.DataFrame([
            {**{'building':b.name}, **b.observation_metadata}
            for b in env.buildings
        ]))
        print()
        print('Active actions:')
        display(pd.DataFrame([
            {**{'building':b.name}, **b.action_metadata}
            for b in env.buildings
        ]))