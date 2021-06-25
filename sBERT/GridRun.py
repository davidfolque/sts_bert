import torch
import numpy as np
from datetime import datetime
import pandas as pd
import resource
import os.path


def random_sample(n, k, seed):
    return np.random.default_rng(seed=seed).choice(n, k, replace=False).tolist()


def print_stats():
    ru = resource.getrusage(resource.RUSAGE_SELF)
    max_ram_gb = ru.ru_maxrss / 1024 / 1024
    user_time_min = ru.ru_utime / 60

    cuda_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
    cuda_reserved_gb = torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024
    cuda_allocated_gb = torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024

    print('Max RAM used: {:.2f} Gb'.format(max_ram_gb))
    print('User time: {:.2f} min'.format(user_time_min))
    print('GPU usage: total {:.2f} Gb, reserved {:.2f}, allocated {:.2f}'.format(cuda_total_gb,
        cuda_reserved_gb, cuda_allocated_gb))


def construct_configs(grid):
    configs = [{}]
    for key, values in grid.items():
        if type(values) != list:
            values = [values]
        new_configs = []
        for value in values:
            new_configs += [{**con, key: value} for con in configs]
        configs = new_configs
    return configs


def grid_run(grid, run_experiment_fnc, load_path=None, save_dir='.'):
    if not os.path.isdir(save_dir):
        print('Error: save path {} is not a valid directory'.format(save_dir))
        return

    time_string = datetime.now().strftime("%y%m%d_%H%M%S")
    if load_path is None:
        df_results = pd.DataFrame(columns=list(grid.keys()) + ['test_score'])
    else:
        df_results = pd.read_csv(load_path)

    for config in construct_configs(grid):
        previous_result = df_results.merge(pd.DataFrame([config]))
        if len(previous_result) > 0:
            print('Already done: ' + str(previous_result.to_dict('list')))
            continue

        print('Run experiment ' + str(config))
        result = run_experiment_fnc(config)[0]
        print(config)
        print('Test score: {:.4f}'.format(result))
        print_stats()
        df_results = df_results.append({**config, 'test_score': result}, ignore_index=True)

        df_results.to_csv(save_dir + '/results_' + time_string + '.csv', index=False)
        df_results.to_csv(save_dir + '/backup/results_' + time_string + '.csv', index=False)

    return df_results
