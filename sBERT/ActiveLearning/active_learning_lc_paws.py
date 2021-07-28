
from Datasets import load_paws_wiki
from CrossEncoder import CrossEncoder, CrossEncoderPretrained
from PawsTrainer import PawsTrainer
import torch
import numpy as np
from ActiveLearning.ALDataLoader import ALDataLoader
from GridRun import GridRun, get_array_info
import pickle


datasets_path = '/home/cs-folq1/rds/rds-t2-cspp025-5bF3aEHVmLU/cs-folq1/datasets/'
# datasets_path = '/home/david/Documents/PhD/rds/datasets/'

paws_dataset = load_paws_wiki(datasets_path=datasets_path)

if False:
    print(len(paws_dataset['train']))
    paws_dataset['train'] = torch.utils.data.Subset(paws_dataset['train'], range(15000))
    paws_dataset['dev'] = torch.utils.data.Subset(paws_dataset['dev'], range(2000))
    paws_dataset['test'] = torch.utils.data.Subset(paws_dataset['test'], range(5000))
    print(len(paws_dataset['train']))


# original_model = CrossEncoderPretrained(CrossEncoder(mode='nli-base'), mode='as-is')
# original_model = None

array_info = get_array_info()

grid_run = GridRun(None, 'al_lc_paws_1000_500_3500_random_starts', array_info=array_info)
experiment_dir = grid_run.persistence.experiment_dir


def run_experiment(config):
    train_dl = ALDataLoader(paws_dataset['train'], batch_size=config['batch_size'],
                            shuffle_train=True, seed=config['train_subset_seed'])
    train_dl.select_k_at_random(config['initial_k'])
    dev_dl = torch.utils.data.DataLoader(paws_dataset['dev'], batch_size=config['batch_size'],
                                         shuffle=False)
    test_dl = torch.utils.data.DataLoader(paws_dataset['test'], batch_size=config['batch_size'],
                                          shuffle=False)

    training_progresses = []
    dev_performances = []
    all_confidences = []
    all_indices = [np.nonzero(train_dl.selected)[0]]

    for i in range(config['times'] + 1):
        train_dl.train()

        assert config['encoder'] == 'cross' and config['pretrained_model'] == 'nli'
        model = CrossEncoderPretrained(CrossEncoder(mode='nli-base'), mode='as-is')
        # if original_model is not None:
        #     model.load_state_dict(original_model.state_dict())
        trainer = PawsTrainer(model=model, train_dl=train_dl, dev_dl=dev_dl, test_dl=test_dl,
                              num_epochs=config['n_epochs'], batch_size=config['batch_size'],
                              lr=config['lr'])

        trainer.train(disable_progress_bar=True, verbose=False, eval_zero_shot=True,
                      early_stopping=False)
        training_progresses.append(trainer.training_progress)
        dev_performances.append(trainer.best_dev_performance)
        test_score, test_loss = trainer.score(trainer.test_dl, disable_progress_bar=False)
        print('Dataloader size: {}, dev score: {}, test score: {}'.format(
            np.sum(train_dl.selected), trainer.best_dev_performance, test_score))

        if i < config['times']:
            if config['mode'] == 'rnd':
                old_selected = train_dl.selected.copy()
                train_dl.select_k_at_random(config['k'])
                all_indices.append(np.nonzero(train_dl.selected & ~old_selected)[0])
            else:
                train_dl.selection()
                probs, _ = trainer.predict(train_dl, disable_progress_bar=False)
                probs = np.expand_dims(np.array(probs), 1)
                confidences = np.concatenate((probs, 1 - probs), axis=1).max(axis=1)
                all_confidences.append(confidences)
                if config['mode'] == 'mc':
                    confidences = -confidences
                else:
                    assert config['mode'] == 'lc'
                lc_indices = np.argpartition(confidences, range(config['k']))[:config['k']]
                all_indices.append(train_dl.selection_indices[lc_indices])
                train_dl.select_indices(train_dl.selection_indices[lc_indices])

    contents = {
        'training_progresses': training_progresses,
        'dev_performances': dev_performances,
        'all_confidences': all_confidences,
        'all_indices': all_indices
    }
    contents_file_name = experiment_dir + '/contents_' + config['mode'] + '_' + \
                         config['train_subset_seed'] + '.pickle'
    with open(contents_file_name, 'wb') as f:
        pickle.dump(contents, f)

    return test_score, None, None


grid = {
    'mode': ['lc', 'mc', 'rnd'],
    'initial_k': 1000,
    'k': 500,
    'times': 5,
    'n_epochs': 5,
    'batch_size': 50,
    'lr': 5e-5,
    'encoder': 'cross',
    'pretrained_model': 'nli',
    'train_subset_seed': 1
}

if array_info is not None:
    grid['mode'] = grid['mode'][array_info.task_id]

grid_run.run_experiment_fnc = run_experiment
grid_run.run(grid)
