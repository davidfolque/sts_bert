
from Datasets import load_wiki_qa
from CrossEncoder import CrossEncoder, CrossEncoderPretrained
from WikiQA.WikiQABinaryClassifierTrainer import WikiQABinaryClassifierForALTrainer
import torch
import numpy as np
from GridRun import GridRun, get_array_info
from ActiveLearning.ActiveLearningRun import ActiveLearningRun
import pickle

wikiqa_dataset = load_wiki_qa()

if False:
    assert False
    print(len(wikiqa_dataset['validation']))
    # wikiqa_dataset['train'] = torch.utils.data.Subset(wikiqa_dataset['train'], range(15000))
    wikiqa_dataset['validation'] = \
        torch.utils.data.Subset(wikiqa_dataset['validation'], range(1000))
    # wikiqa_dataset['test'] = torch.utils.data.Subset(wikiqa_dataset['test'], range(5000))
    print(len(wikiqa_dataset['validation']))


class ALRunForWikiQA(ActiveLearningRun):
    # Override
    def create_model(self):
        original_sts_cross_model_path = 'results/pretraining_sts/results_210702_094435_' \
                                        'cross-encoder_cls-pooling-hidden_best_model.bin'
        max_length_answer = 128
        sts_model = CrossEncoder(mode='cls-pooling-hidden', max_length_second=2 * max_length_answer)
        sts_model.load_state_dict(torch.load(original_sts_cross_model_path))
        wiki_qa_model = CrossEncoderPretrained(sts_model, mode='replace-head')
        return wiki_qa_model

    # Override
    def create_trainer(self, model, train_dl, dev_dl, test_dl, num_epochs, lr):
        return WikiQABinaryClassifierForALTrainer(model=model, train_dl=train_dl, dev_dl=dev_dl,
                                                  test_dl=test_dl, mode='scaled',
                                                  num_epochs=num_epochs, lr=lr)

    # Override
    def tokenize_batch(self, model, batch):
        return model.tokenize(batch['question'], batch['answer'])


array_info = get_array_info()

grid_run = GridRun(None, 'al_lc_paws_100_50_350_random_starts', array_info=array_info)
experiment_dir = grid_run.persistence.experiment_dir


def run_experiment(config):
    al_run = ALRunForWikiQA(wikiqa_dataset['train'], wikiqa_dataset['validation'],
                            wikiqa_dataset['test'])

    al_run.run(mode=config['mode'], initial_k=config['initial_k'], inc_k=config['k'],
               times=config['times'], n_epochs=config['n_epochs'], batch_size=config['batch_size'],
               lr=config['lr'], train_subset_seed=config['train_subset_seed'], min_positives=5)

    contents = {
        'training_progresses': al_run.training_progresses,
        'dev_scores': al_run.dev_scores,
        'test_scores': al_run.test_scores,
        'all_probs': al_run.all_probs,
        'all_indices': al_run.all_indices
    }
    contents_file_name = experiment_dir + '/contents_' + config['mode'] + '_' + \
                         str(config['train_subset_seed']) + '.pickle'
    with open(contents_file_name, 'wb') as f:
        pickle.dump(contents, f)

    # This is hacky. We are not interested in final score or loss anyway.
    return (al_run.test_scores[-1], al_run.test_scores[-1]), None, None


grid = {
    'mode': ['lc', 'mc', 'rnd', 'lc_kmeans-mean', 'lc_kmeans-cls'],
    'initial_k': 100,
    'k': 50,
    'times': 5,
    'n_epochs': 5,
    'batch_size': 32,
    'lr': 2e-5,
    'encoder': 'cross',
    'pretrained_model': 'sts',
    'train_subset_seed': [1, 2, 3, 4, 5]
}

if array_info is not None:
    grid['train_subset_seed'] = grid['train_subset_seed'][array_info.task_id]

grid_run.run_experiment_fnc = run_experiment
grid_run.run(grid)































