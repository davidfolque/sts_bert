import torch
from datetime import datetime
import pandas as pd
import os.path
import re
from filelock import FileLock
from collections import namedtuple


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

        self.results = load_last_results_from_disk(self.experiment_dir)
        if self.results is None:
            self.results = pd.DataFrame()

        self.save_file_name = None

    @staticmethod
    def create_save_file_name(execution_name):
        time_string = datetime.now().strftime("%y%m%d_%H%M%S")
        save_file_name = 'results_' + time_string
        if self.execution_name is not None:
            save_file_name += '_' + execution_name
        save_file_name += '.csv'
        return save_file_name

    # Recovers the save_file_name. If none is found, creates a new one and stores it.
    def set_save_file_name(self):
        if self.array_info is None:
            self.save_file_name = create_save_file_name(self.execution_name)
        else:
            self.locks_dir = self.experiment_dir + 'locks/'
            check_create_dir(self.locks_dir)
            array_job_file = self.locks_dir + '/array_job_{}.txt'.format(self.array_info.array_id)
            with FileLock(array_job_file + '.lock'):
                if os.path.isfile(array_job_file):
                    with open(array_job_file, 'r') as file:
                        self.save_file_name = file.read()
                else:
                    self.save_file_name = create_save_file_name(self.execution_name)
                    with open(array_job_file, 'w') as file:
                        file.write(self.save_file_name)
        print('Results will be stored in file ' + self.save_file_name)

    def save_results(self, results):
        if self.save_file_name is None:
            self.set_save_file_name()

        self.df_results.to_csv(self.backup_dir + self.save_file_name, index=False)
        if self.array_info is None:
            self.df_results.to_csv(self.experiment_dir + self.save_file_name, index=False)
        else:
            with FileLock(self.experiment_dir + self.save_file_name + '.lock'):
                try:
                    print('Lock acquired.')
                    self.df_results.to_csv(self.experiment_dir + self.save_file_name,
                                           index=False)
                finally:
                    print('Unlocking lock.')

    def save_model(self, model, model_name):
        final_file_name = self.experiment_dir + self.save_file_name[:-4] + '_' + model_name + \
                          '_best_model.bin'
        print('Best {} model stored in file {}'.format(model_name, final_file_name))
        torch.save(model.state_dict(), final_file_name)