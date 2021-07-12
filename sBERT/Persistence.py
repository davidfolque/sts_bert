import torch
from datetime import datetime
import pandas as pd
import os.path
import re
from filelock import FileLock
from collections import namedtuple


class DeactivableLock:
    def __init__(self, file, disable=False):
        self.lock = None if disable else FileLock(file)
        self.file = file

    def __enter__(self):
        if self.lock:
            print('Aqcuiring lock for file ' + self.file)
            self.lock.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.lock:
            print('Releasing lock for file ' + self.file)
            self.lock.release()


def check_create_dir(path):
    if not os.path.isdir(path):
        print('Creating dir ' + path)
        os.mkdir(path)


def load_last_results_from_disk(dir_path, current_file=None):
    results_files = [filename for filename in os.listdir(dir_path) if
                     re.match(r'^results_\d{6}_\d{6}.*\.csv$',
                              filename) and filename != current_file]

    if len(results_files) == 0:
        print('Did not load any previous results')
        return None

    results_files.sort()
    load_path = dir_path + '/' + results_files[-1]
    print('Loading previous results from ' + load_path)
    return pd.read_csv(load_path)


ArrayJobInfo = namedtuple('ArrayJobInfo', 'array_id task_id')

class Persistence:
    def __init__(self, experiment_name, execution_name, array_info, root_dir):
        self.execution_name = execution_name
        self.array_info = array_info

        if not os.path.isdir(root_dir):
            raise Exception('root dir {} is not a valid directory'.format(root_dir))

        self.experiment_dir = root_dir + '/' + experiment_name + '/'
        check_create_dir(self.experiment_dir)

        if self.array_info is None:
            self.backup_dir = self.experiment_dir + 'backup/'
        else:
            self.backup_dir = self.experiment_dir + 'backup_{}/'.format(self.array_info.task_id)
        check_create_dir(self.backup_dir)

        self.set_save_file_name()
        
    def load_results(self):
        results = load_last_results_from_disk(self.experiment_dir, current_file=self.save_file_name)
        if results is None:
            results = pd.DataFrame()
        return results


    @staticmethod
    def create_save_file_name(execution_name):
        time_string = datetime.now().strftime("%y%m%d_%H%M%S")
        save_file_name = 'results_' + time_string
        if execution_name is not None:
            save_file_name += '_' + execution_name
        save_file_name += '.csv'
        return save_file_name

    # Recovers the save_file_name. If none is found, creates a new one and stores it.
    def set_save_file_name(self):
        if self.array_info is None:
            self.save_file_name = self.create_save_file_name(self.execution_name)
        else:
            self.locks_dir = self.experiment_dir + 'locks/'
            check_create_dir(self.locks_dir)
            array_job_file = self.locks_dir + '/array_job_{}.txt'.format(self.array_info.array_id)
            with FileLock(array_job_file + '.lock'):
                if os.path.isfile(array_job_file):
                    with open(array_job_file, 'r') as file:
                        self.save_file_name = file.read()
                else:
                    self.save_file_name = self.create_save_file_name(self.execution_name)
                    with open(array_job_file, 'w') as file:
                        file.write(self.save_file_name)
        print('Results will be stored in file ' + self.save_file_name)


    @staticmethod
    def append_result_inner(dict_result, file_name):
        if os.path.isfile(file_name):
            results = pd.read_csv(file_name)
        else:
            results = pd.DataFrame()
        results.append(dict_result, ignore_index=True)
        results.to_csv(backup_file_name, index=False)
        return results

    def append_result(self, dict_result):
        # Read, update and write backup.
        self.append_result_inner(dict_result, self.backup_dir + self.save_file_name)

        # Read, update and write data, possibly taking a lock.
        save_file_name = self.experiment_dir + self.save_file_name
        with DeactivableLock(save_file_name + '.lock', disable=self.array_info is None):
            return self.append_result_inner(dict_result, save_file_name)



    def save_model(self, model, model_name):
        final_file_name = self.experiment_dir + self.save_file_name[:-4] + '_' + model_name + \
                          '_best_model.bin'
        print('Best {} model stored in file {}'.format(model_name, final_file_name))
        torch.save(model.state_dict(), final_file_name)
