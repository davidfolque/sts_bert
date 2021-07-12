import torch
import numpy as np
from datetime import datetime
import pandas as pd
import resource
import os.path
import re
from filelock import FileLock


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

def load_last_results_from_disk(dir_path):
    results_files = [filename[8:-4] for filename in os.listdir(dir_path) if
                     re.match(r'^results_\d{6}_\d{6}.*\.csv$', filename)]

    if len(results_files) == 0:
        print('Did not load any previous results')
        return None

    results_files.sort()
    load_path = dir_path + '/results_' + results_files[-1] + '.csv'
    print('Loading previous results from ' + load_path)
    return pd.read_csv(load_path)


class GridRun():
    def __init__(self, run_experiment_fnc, results_dir=None, experiment_name=None, task_id=None):

        self.task_id = task_id

        if results_dir is None or experiment_name is None:
            print('Either results_dir or experiment_name is None: won\'t store the results')
            self.experiment_dir = None
            self.df_results = pd.DataFrame()
        elif not os.path.isdir(results_dir):
            raise Exception('results dir {} is not a valid directory'.format(results_dir))
        else:
            self.experiment_dir = results_dir + '/' + experiment_name + '/'
            check_create_dir(self.experiment_dir)

            if self.task_id is None:
                self.backup_dir = self.experiment_dir + 'backup/'
            else:
                self.backup_dir = self.experiment_dir + 'backup_{}/'.format(task_id)
            check_create_dir(self.backup_dir)

            self.df_results = load_last_results_from_disk(self.experiment_dir)
            if self.df_results is None:
                self.df_results = pd.DataFrame()

        self.run_experiment_fnc = run_experiment_fnc

    def run(self, grid, exec_name=None, ignore_previous_results=False, save_best=False):
        save_file_name = None
        if self.experiment_dir is not None:
            time_string = datetime.now().strftime("%y%m%d_%H%M%S")
            save_file_name = 'results_' + time_string
            if exec_name is not None:
                save_file_name += '_' + exec_name
            print('Results will be stored in file ' + save_file_name)
        else:
            if exec_name is not None:
                raise Exception('exec_name is set but results_dir and experiment_name are not.')
            if save_best:
                raise Exception('save_best is True but results_dir and experiment_name are None.')
            print('Results won\'t be saved')

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
            score_and_loss, model, save_name = self.run_experiment_fnc(config)
            result = score_and_loss[0]
            print('Test score: {:.4f}'.format(result))
            print_stats()

            if save_best and (save_name not in best_scores or best_scores[save_name] < result):
                best_scores[save_name] = result
                best_model_file_name = self.experiment_dir + save_file_name + '_' + save_name + \
                                       '_best_model.bin'
                print('Best {} model stored in file {}'.format(save_name, best_model_file_name))
                torch.save(model.state_dict(), best_model_file_name)

            self.df_results = self.df_results.append({**config, 'test_score': result},
                                                     ignore_index=True)
            del model
            torch.cuda.empty_cache()

            if save_file_name is not None:
                self.df_results.to_csv(self.backup_dir + save_file_name + '.csv', index=False)
                with FileLock(self.experiment_dir + save_file_name + '.csv.lock'):
                    try:
                        print('Lock acquired.')
                        self.df_results.to_csv(self.experiment_dir + save_file_name + '.csv',
                                               index=False)
                    finally:
                        print('Unlocking lock.')
