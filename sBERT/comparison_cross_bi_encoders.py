import torch
from torch.utils.data import DataLoader

from BiEncoder import BiEncoder
from CrossEncoder import CrossEncoder
from STSTrainer import STSTrainer
from Datasets import load_sts
from GridRun import grid_run, random_sample


dataset = load_sts()


def run_experiment(config):
    subset_indices = random_sample(n=len(dataset['train']), k=config['train_size'],
                                   seed=config['train_subset_seed'])
    train_dataset_subset = torch.utils.data.Subset(dataset['train'], subset_indices)

    encoder, mode = config['mode'].split('/')
    encoder_class = BiEncoder if encoder == 'bi-encoder' else CrossEncoder
    model = encoder_class(mode=mode)

    trainer = STSTrainer(model=model, train_dataset=train_dataset_subset, dataset=dataset,
                         num_epochs=config['num_epochs'], batch_size=config['batch_size'],
                         lr=config['lr'], lr_scheduler=config['lr_scheduler'],
                         warmup_percent=config['warmup_percent'])
    result = trainer.train(disable_progress_bar=True)
    del model
    torch.cuda.empty_cache()
    return result


grid = {
    'num_epochs': 10,
    'batch_size': 16,
    'lr': 2e-5,
    'lr_scheduler': 'linear',
    'warmup_percent': 0.2,
    'mode': ['bi-encoder/base-linear-pooling',
             'bi-encoder/base-mean-pooling',
             'bi-encoder/nli-linear-pooling',
             'bi-encoder/nli-mean-pooling',
             'cross-encoder/nli-base',
             'cross-encoder/cls-pooling-hidden'],
    'train_size': [500, 1000, 2000, 3000, 4000, len(dataset['train'])],
    'train_subset_seed': [1, 2, 3]
}

df_results = grid_run(grid, run_experiment,
                      load_path='./comparison_results/results_210621_172128.csv',
                      save_dir='./comparison_results')

