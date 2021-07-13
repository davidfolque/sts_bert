import torch
import numpy as np
from datetime import datetime
import pandas as pd
import resource
import os.path
import re
from filelock import FileLock
from Persistence import Persistence, ArrayJobInfo


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


def construct_query(d):
    query = ''
    for key, value in d.items():
        query += key + '=='
        if type(value) == str:
            query += '"' + value + '"'
        else:
            query += str(value)
        query += ' and '
    return query[:-5]


class GridRun():
    def __init__(self, run_experiment_fnc, experiment_name, execution_name=None, array_info=None,
                 root_dir='./results'):

        self.run_experiment_fnc = run_experiment_fnc

        if experiment_name is None:
            print('experiment_name is None: won\'t store the results')
            self.persistence = None
            self.df_results = pd.DataFrame()
        else:
            self.persistence = Persistence(experiment_name=experiment_name,
                                           execution_name=execution_name, array_info=array_info,
                                           root_dir=root_dir)
            self.df_results = self.persistence.init_and_load_previous_results()

    def run(self, grid, ignore_previous_results=False, save_best=False):
        best_scores = {}
        for config in construct_configs(grid):
            if len(self.df_results) > 0:
                query = construct_query(config)
                previous_result = self.df_results.query(query)
                if len(previous_result) > 0:
                    print('Already done: ' + str(previous_result.to_dict('list')))
                    if ignore_previous_results:
                        print('Repeating experiment')
                        self.df_results.query('not (' + query + ')', inplace=True)
                    else:
                        print('Skipping experiment')
                        continue

            print('----------\nRUN CONFIG\n----------')
            for key, value in config.items():
                print(key, ': ', value)
            score_and_loss, model, model_save_name = self.run_experiment_fnc(config)
            result = score_and_loss[0]
            print('Test score: {:.4f}'.format(result))
            print_stats()

            # Store results.
            if self.persistence is not None:
                self.df_results = self.persistence.append_result({**config, 'test_score': result})

            # Update and store best model.
            if save_best and (model_save_name not in best_scores or \
                              best_scores[model_save_name] < result):
                best_scores[model_save_name] = result
                self.persistence.save_model(model, model_save_name)

            del model
            torch.cuda.empty_cache()

