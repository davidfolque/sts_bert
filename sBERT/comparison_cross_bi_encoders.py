import torch
from torch.utils.data import DataLoader

from BiEncoder import BiEncoder
from CrossEncoder import CrossEncoder
from STSTrainer import STSTrainer
from Datasets import load_sts
from GridRun import GridRun, random_sample

dataset = load_sts()


def run_experiment(config):
    subset_indices = random_sample(n=len(dataset['train']), k=config['train_size'],
                                   seed=config['train_subset_seed'])
    train_dataset_subset = torch.utils.data.Subset(dataset['train'], subset_indices)

    encoder, mode = config['mode'].split('/')
    if encoder == 'bi-encoder':
        model = BiEncoder(mode=mode, head='cos-sim')
    else:
        model = CrossEncoder(mode=mode)

    trainer = STSTrainer(model=model, train_dataset=train_dataset_subset, dataset=dataset,
                         num_epochs=config['num_epochs'], batch_size=config['batch_size'],
                         lr=config['lr'], lr_scheduler=config['lr_scheduler'],
                         warmup_percent=config['warmup_percent'])
    result = trainer.train(disable_progress_bar=True)
    save_name = encoder + '_' + mode
    return result, model, save_name


grid = {
    'num_epochs': 10,
    'batch_size': 16,
    'lr': 2e-5,
    'lr_scheduler': 'linear',
    'warmup_percent': 0.2,
    'mode': [
             'bi-encoder/nli-cls-pooling',
             'bi-encoder/nli-mean-pooling',
             'bi-encoder/nli-linear-pooling'],
    #'train_size': [500, 1000, 2000, 3000, 4000, len(dataset['train'])],
    'train_size': [len(dataset['train'])],
    'train_subset_seed': [1, 2, 3]
}

grid_run = GridRun(run_experiment, results_dir='results', experiment_name='pretraining_sts')
grid_run.run(grid, save_best=True, ignore_previous_results=True)
# grid_run.df_results

