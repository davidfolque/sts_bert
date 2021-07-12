import torch
import os
from Datasets import load_wiki_qa
from WikiQA.WikiQAClassifierTrainer import WikiQAClassifierTrainer
from WikiQA.WikiQABinaryClassifierTrainer import WikiQABinaryClassifierTrainer
from CrossEncoder import CrossEncoder, CrossEncoderPretrained
from BiEncoder import BiEncoder
from GridRun import GridRun

wiki_qa = load_wiki_qa()

original_sts_cross_model_path = \
    'results/pretraining_sts/results_210702_094435_cross-encoder_cls-pooling-hidden_best_model.bin'
original_nli_sts_cross_model_path = \
    'results/pretraining_sts/results_210702_094435_cross-encoder_nli-base_best_model.bin'
original_sts_bi_model_path = \
    'results/pretraining_sts/results_210707_152607_bi-encoder_base-cls-pooling_best_model.bin'
original_nli_sts_bi_model_path = \
    'results/pretraining_sts/results_210707_152607_bi-encoder_nli-cls-pooling_best_model.bin'

#%%

# Train on WikiQA.

def run_experiment(config):
    # encoder, pretrained_model = config['pretrained_model'].split('/')
    encoder = config['encoder']
    pretrained_model = config['pretrained_model']

    if encoder == 'cross':
        if pretrained_model in ['bert', 'sts']:
            sts_model = CrossEncoder(mode='cls-pooling-hidden')
            if pretrained_model == 'sts':
                print('Loading sts pretrained model from ' + original_sts_cross_model_path)
                sts_model.load_state_dict(torch.load(original_sts_cross_model_path))
        else:
            assert pretrained_model in ['nli', 'nli-sts']
            sts_model = CrossEncoder(mode='nli-base')
            if pretrained_model == 'nli-sts':
                print('Loading nli and sts pretrained model from ' +
                    original_nli_sts_cross_model_path)
                sts_model.load_state_dict(torch.load(original_nli_sts_cross_model_path))

        wiki_qa_model = CrossEncoderPretrained(sts_model, mode='replace-head')
        # if pretrained_model in ['bert', 'nli']:
        #     wiki_qa_model = CrossEncoderPretrained(sts_model, mode='as-is')
        # else:
        #     assert(pretrained_model in ['sts', 'nli-sts'])
        #     wiki_qa_model = CrossEncoderPretrained(sts_model, mode='replace-head')
    else:
        assert (encoder == 'bi')
        if pretrained_model == 'nli':
            wiki_qa_model = BiEncoder(mode='nli-cls-pooling', head='extra-head-sub')
        else:
            wiki_qa_model = BiEncoder(mode='base-cls-pooling', head='extra-head-sub')
            if pretrained_model != 'bert':
                if pretrained_model == 'nli-sts':
                    model_path = original_nli_sts_bi_model_path
                else:
                    assert pretrained_model == 'sts'
                    model_path = original_sts_bi_model_path
                print('Loading pretrained model from ' + model_path)
                load_result = wiki_qa_model.load_state_dict(torch.load(model_path), strict=False)

                # Assert that the only keys missing are extra head.
                assert (load_result.missing_keys == ['extra_head.weight', 'extra_head.bias'])
                assert (load_result.unexpected_keys == [])

    trainer = WikiQABinaryClassifierTrainer(model=wiki_qa_model, dataset=wiki_qa,
                                            mode=config['sample_selection'],
                                            trainset_size=config['train_size'],
                                            trainset_seed=config['train_subset_seed'],
                                            num_epochs=config['num_epochs'],
                                            batch_size=config['batch_size'], lr=config['lr'],
                                            devset_size=max(1000, config['train_size']))
    result = trainer.train(disable_progress_bar=False, eval_zero_shot=False, early_stopping=True)
    save_name = 'pretrained_' + config['pretrained_model']
    return result, wiki_qa_model, save_name


grid = {
    'num_epochs': 10,  # Size 5000 => 50s per epoch????
    'batch_size': 16,
    'lr': 2e-5,
    'encoder': ['cross', 'bi'],
    'pretrained_model': [
        'bert', 'nli', 'sts', 'nli-sts'],
    'sample_selection': ['downsampling', 'all-scaled', 'all-unscaled'],
    # 'mode': ['replace-head', 'shift-bias', 'additional-head'],
    # 'mode': 'shift-bias',
    # 'train_size': 2000,
    #'train_size': [10, 50, 100, 500, 1000, 2000, 5000, 10000],
    'train_size': 2000,
    'train_subset_seed': [1, 2, 3]
}


n_tasks = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
if n_tasks > 1:
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    grid['pretrained_model'] = grid['pretrained_model'][task_id]

grid_run = GridRun(run_experiment, results_dir='results',
                   experiment_name='few_shot_wikiqa_{}'.format(task_id))
grid_run.run(grid, save_best=False, exec_name='only_2000')





