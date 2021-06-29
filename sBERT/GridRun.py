import torch
import numpy as np
from datetime import datetime
import pandas as pd
import resource
import os.path
import re


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


def check_create_dir(path):
    if not os.path.isdir(path):
        print('Creating dir ' + path)
        os.mkdir(path)


def grid_run(grid, run_experiment_fnc, results_dir=None, exec_name=None, save_name=None):
    if not os.path.isdir(results_dir):
        print('Error: results dir {} is not a valid directory'.format(results_dir))
        return
    exec_dir = results_dir + '/' + exec_name
    check_create_dir(exec_dir)
    check_create_dir(exec_dir + '/backup')

    results_files = [filename[8:21] for filename in os.listdir(exec_dir)
                     if re.match(r'^results_\d{6}_\d{6}.*\.csv$', filename)]

    if len(results_files) == 0:
        print('Did not load any previous results')
        df_results = pd.DataFrame(columns=list(grid.keys()) + ['test_score'])
    else:
        results_files.sort()
        load_path = exec_dir + '/results_' + results_files[-1] + '.csv'
        print('Loading previous results from ' + load_path)
        df_results = pd.read_csv(load_path)

    time_string = datetime.now().strftime("%y%m%d_%H%M%S")
    final_save_name = '/results_' + time_string
    if save_name is not None:
        final_save_name += '_' + save_name
    final_save_name += '.csv'
    print('Results will be stored in file ' + final_save_name)

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

        df_results.to_csv(exec_dir + final_save_name, index=False)
        df_results.to_csv(exec_dir + '/backup' + final_save_name, index=False)

    return df_results
