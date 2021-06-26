import torch

from Datasets import load_sts, load_paws_wiki
from CrossEncoder import CrossEncoder, CrossEncoderPretrained
from STSTrainer import STSTrainer
from PawsTrainer import PawsTrainer
from GridRun import grid_run, random_sample

# Load datasets.

datasets_path = '/home/cs-folq1/rds/rds-t2-cspp025-5bF3aEHVmLU/cs-folq1/datasets/'
# datasets_path = '/home/david/Documents/PhD/rds/datasets/'
sts_dataset = load_sts()
paws_dataset = load_paws_wiki(datasets_path=datasets_path)


# Train original cross encoder.

sts_model = CrossEncoder(mode='cls-pooling-hidden')
sts_trainer = STSTrainer(model=sts_model, train_dataset=sts_dataset['train'],
                         dataset=sts_dataset, num_epochs=10, lr_scheduler='linear',
                         warmup_percent=0.2)
# sts_trainer.train(disable_progress_bar=True)
original_sts_model_path = './saved_models/few_shot_paws/original_cross_encoder.bin'
# torch.save(sts_model.state_dict(), original_sts_model_path)


# Train on PAWS.

def run_experiment(config):
    subset_indices = random_sample(n=len(paws_dataset['train']), k=config['train_size'],
                                   seed=config['train_subset_seed'])
    train_dataset_subset = torch.utils.data.Subset(paws_dataset['train'], subset_indices)

    sts_model.load_state_dict(torch.load(original_sts_model_path))
    paws_model = CrossEncoderPretrained(sts_model, mode=config['mode'])

    trainer = PawsTrainer(model=paws_model, train_dataset=train_dataset_subset,
                          dataset=paws_dataset, num_epochs=config['num_epochs'],
                          batch_size=config['batch_size'], lr=config['lr'])
    result = trainer.train(disable_progress_bar=True, eval_zero_shot=True)
    del paws_model
    torch.cuda.empty_cache()
    return result


grid = {
    'num_epochs': 5,
    'batch_size': 16,
    'lr': 2e-5,
    'mode': ['replace-head', 'shift-bias', 'additional-head'],
    'train_size': [50, 100, 200, 500, 1000, 2000, 10000, len(paws_dataset['train'])],
    'train_subset_seed': [1, 2, 3]
}

df_results = grid_run(grid, run_experiment, load_path=None, save_dir='./comparison_results')
