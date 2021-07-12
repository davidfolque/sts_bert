import torch
import os
from Datasets import load_sts, load_paws_wiki
from CrossEncoder import CrossEncoder, CrossEncoderPretrained
from BiEncoder import BiEncoder
from PawsTrainer import PawsTrainer
from GridRun import GridRun, random_sample


# Load datasets.

datasets_path = '/home/cs-folq1/rds/rds-t2-cspp025-5bF3aEHVmLU/cs-folq1/datasets/'
# datasets_path = '/home/david/Documents/PhD/rds/datasets/'
sts_dataset = load_sts()
paws_dataset = load_paws_wiki(datasets_path=datasets_path)



original_sts_cross_model_path = \
    'results/pretraining_sts/results_210702_094435_cross-encoder_cls-pooling-hidden_best_model.bin'
original_nli_sts_cross_model_path = \
    'results/pretraining_sts/results_210702_094435_cross-encoder_nli-base_best_model.bin'
original_sts_bi_model_path = \
    'results/pretraining_sts/results_210707_152607_bi-encoder_base-cls-pooling_best_model.bin'
original_nli_sts_bi_model_path = \
    'results/pretraining_sts/results_210708_103406_bi-encoder_nli-cls-pooling_best_model.bin'


# Train on PAWS.

def run_experiment(config):
    subset_indices = random_sample(n=len(paws_dataset['train']), k=config['train_size'],
                                   seed=config['train_subset_seed'])
    train_dataset_subset = torch.utils.data.Subset(paws_dataset['train'], subset_indices)

    if config['train_size'] < len(paws_dataset['dev']):
        dev_subset_indices = random_sample(n=len(paws_dataset['dev']),
                                           k=max(config['train_size'], 500),
                                           seed=config['train_subset_seed'])
        dev_dataset_subset = torch.utils.data.Subset(paws_dataset['dev'], dev_subset_indices)
    else:
        dev_dataset_subset = paws_dataset['dev']

    encoder, pretrained_model = config['pretrained_model'].split('/')

    if encoder == 'cross':
        if pretrained_model in ['bert', 'sts']:
            sts_model = CrossEncoder(mode='cls-pooling-hidden')
            if pretrained_model == 'sts':
                print('Loading sts pretrained model from ' + original_sts_cross_model_path)
                sts_model.load_state_dict(torch.load(original_sts_cross_model_path))
        elif pretrained_model in ['nli', 'nli-sts']:
            sts_model = CrossEncoder(mode='nli-base')
            if pretrained_model == 'nli-sts':
                print(
                    'Loading nli and sts pretrained model from ' +
                    original_nli_sts_cross_model_path)
                sts_model.load_state_dict(torch.load(original_nli_sts_cross_model_path))
        else:
            assert (pretrained_model == 'nli-3rd-component')
            sts_model = CrossEncoder(mode='nli-head-3rd-component')

        if pretrained_model in ['bert', 'nli', 'nli-3rd-component']:
            paws_model = CrossEncoderPretrained(sts_model, mode='as-is')
        else:
            assert (pretrained_model in ['sts', 'nli-sts'])
            paws_model = CrossEncoderPretrained(sts_model, mode='replace-head')
    else:
        assert (encoder == 'bi')
        if pretrained_model == 'nli':
            paws_model = BiEncoder(mode='nli-cls-pooling', head='extra-head-sub')
        else:
            paws_model = BiEncoder(mode='base-cls-pooling', head='extra-head-sub')
            if pretrained_model != 'bert':
                if pretrained_model == 'nli-sts':
                    model_path = original_nli_sts_bi_model_path
                else:
                    assert pretrained_model == 'sts'
                    model_path = original_sts_bi_model_path
                print('Loading pretrained model from ' + model_path)
                load_result = paws_model.load_state_dict(torch.load(model_path), strict=False)

                # Assert that the only keys missing are extra head.
                assert (load_result.missing_keys == ['extra_head.weight', 'extra_head.bias'])
                assert (load_result.unexpected_keys == [])

    trainer = PawsTrainer(model=paws_model, train_dataset=train_dataset_subset,
                          dataset={'dev': dev_dataset_subset, 'test': paws_dataset['test']},
                          num_epochs=config['num_epochs'], batch_size=config['batch_size'],
                          lr=config['lr'])
    result = trainer.train(disable_progress_bar=True, eval_zero_shot=False, early_stopping=True)
    save_name = 'pretrained_' + config['pretrained_model']
    return result, paws_model, save_name


grid = {'num_epochs': 10,  # Size 5000 => 50s per epoch????
        'batch_size': 16, 'lr': 2e-5,
        # 'pretrained_model': ['cross/bert', 'cross/nli-3rd-component', 'cross/sts', 'cross/nli-sts',
        #                      'bi/bert', 'bi/nli', 'bi/sts', 'bi/nli-sts'],
        'pretrained_model': ['cross/bert', 'cross/nli', 'cross/sts', 'cross/nli-sts'],
        # 'mode': ['replace-head', 'shift-bias', 'additional-head'],
        # 'mode': 'shift-bias',
        # 'train_size': [500, 1000, 2000, 5000, 10000],
        'train_size': [10, 50, 100, 500, 1000, 2000, 5000, 10000], 'train_subset_seed': [1, 2, 3]}

n_tasks = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
if n_tasks > 1:
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    grid['pretrained_model'] = grid['pretrained_model'][task_id]

grid_run = GridRun(run_experiment, results_dir='results',
    experiment_name='paws_from_nli_sts_cross_bert_no_temperature_replace_head2_array_{}'.format(
        task_id))
grid_run.run(grid, save_best=False)
