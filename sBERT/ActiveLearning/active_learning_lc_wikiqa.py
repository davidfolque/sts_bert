
from Datasets import load_wiki_qa
from CrossEncoder import CrossEncoder, CrossEncoderPretrained
from WikiQA.WikiQABinaryClassifierTrainer import WikiQABinaryClassifierForALTrainer
import torch
import numpy as np
from ActiveLearning.ALDataLoader import ALDataLoader
from GridRun import GridRun, get_array_info
import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import k_means
from scipy.spatial import KDTree
import gc


wikiqa_dataset = load_wiki_qa()

if False:
    assert False
    print(len(wikiqa_dataset['validation']))
    # wikiqa_dataset['train'] = torch.utils.data.Subset(wikiqa_dataset['train'], range(15000))
    wikiqa_dataset['validation'] = \
        torch.utils.data.Subset(wikiqa_dataset['validation'], range(1000))
    # wikiqa_dataset['test'] = torch.utils.data.Subset(wikiqa_dataset['test'], range(5000))
    print(len(wikiqa_dataset['validation']))


def create_model():
    original_sts_cross_model_path = 'results/pretraining_sts/results_210702_094435_'\
                                    'cross-encoder_cls-pooling-hidden_best_model.bin'
    max_length_answer = 128
    sts_model = CrossEncoder(mode='cls-pooling-hidden', max_length_second=2 * max_length_answer)
    sts_model.load_state_dict(torch.load(original_sts_cross_model_path))
    wiki_qa_model = CrossEncoderPretrained(sts_model, mode='replace-head')
    return wiki_qa_model

# original_model = CrossEncoderPretrained(CrossEncoder(mode='nli-base'), mode='as-is')
# original_model = None

array_info = get_array_info()

grid_run = GridRun(None, 'al_lc_paws_100_50_350_random_starts', array_info=array_info)
experiment_dir = grid_run.persistence.experiment_dir


def run_experiment(config):
    train_dl = ALDataLoader(wikiqa_dataset['train'], batch_size=config['batch_size'],
                            shuffle_train=True, seed=config['train_subset_seed'])
    train_dl.select_k_at_random(config['initial_k'] // 2, ensure=0)
    train_dl.select_k_at_random(config['initial_k'] // 2, ensure=1)
    dev_dl = torch.utils.data.DataLoader(wikiqa_dataset['validation'],
                                         batch_size=config['batch_size'], shuffle=False)
    test_dl = torch.utils.data.DataLoader(wikiqa_dataset['test'], batch_size=config['batch_size'],
                                          shuffle=False)

    training_progresses = []
    dev_performances = []
    test_performances = []
    all_confidences = []
    all_indices = [np.nonzero(train_dl.selected)[0]]

    for i in range(config['times'] + 1):
        train_dl.train()

        assert config['encoder'] == 'cross' and config['pretrained_model'] == 'sts'

        if 'model' in dir():
            del model
        gc.collect()
        torch.cuda.empty_cache()
        model = create_model()

        # if original_model is not None:
        #     model.load_state_dict(original_model.state_dict())

        trainer = WikiQABinaryClassifierForALTrainer(model=model, train_dl=train_dl, dev_dl=dev_dl,
                                                     test_dl=test_dl, mode='scaled',
                                                     num_epochs=config['n_epochs'], lr=config['lr'])
        trainer.train(disable_progress_bar=True, verbose=False, eval_zero_shot=True,
                      early_stopping=False)
        training_progresses.append(trainer.training_progress)
        dev_performances.append(trainer.best_dev_performance)
        test_score, test_loss = trainer.score(trainer.test_dl, disable_progress_bar=False)
        print('Dataloader size: {}, dev score: {:.4f}, test score: {:.4f}'.format(
            np.sum(train_dl.selected), trainer.best_dev_performance, test_score))




        if i < config['times']:
            train_dl.selection()
            probs, _ = trainer.predict(train_dl, disable_progress_bar=False)
            probs = np.expand_dims(np.array(probs), 1)
            confidences = np.concatenate((probs, 1 - probs), axis=1).max(axis=1)
            all_confidences.append(confidences)

            if config['mode'] == 'rnd':
                old_selected = train_dl.selected.copy()
                train_dl.select_k_at_random(config['k'])
                all_indices.append(np.nonzero(train_dl.selected & ~old_selected)[0])
            else:
                if config['mode'] in ['mc', 'lc']:
                    if config['mode'] == 'mc':
                        confidences = -confidences
                    else:
                        assert config['mode'] == 'lc'
                    lc_indices = np.argpartition(confidences, range(config['k']))[:config['k']]
                    query = train_dl.selection_indices[lc_indices]
                else:
                    assert config['mode'] in ['lc_kmeans-mean', 'lc_kmeans-cls']
                    repr_mode = config['mode'].split('-')[1]

                    beta = 10

                    # confidences: confidences of the unlabelled data.
                    #
                    # lc_indices: k*beta least confidences indices, sorted by confidence and
                    # pointing to the unlabelled data.
                    #
                    # representations: representations of the elements of lc_indices, but sorted by
                    # original order (all training data)
                    #
                    # sorted_lc_indices: lc_indices sorted by unlabelled data order => sorted by all
                    # training data order.

                    kbeta = config['k'] * beta
                    lc_indices = np.argpartition(confidences, range(kbeta))[:kbeta]
                    sorted_lc_indices = np.sort(lc_indices)
                    temp_dl = ALDataLoader(wikiqa_dataset['train'], batch_size=config['batch_size'],
                                           shuffle_train=False)
                    temp_dl.select_indices(train_dl.selection_indices[lc_indices])
                    representations = []
                    with torch.no_grad():
                        for batch in temp_dl:
                            inputs = model.tokenize(batch['sentence1'], batch['sentence2'])
                            outputs = model.pretrained_cross_encoder.forward_intermediate(
                                mode=repr_mode, **inputs)
                            representations.append(outputs.cpu().numpy())
                    representations = np.concatenate(representations, axis=0)
                    repr_uncertainties = 2 * (1 - confidences[sorted_lc_indices])

                    centroids = k_means(representations, n_clusters=config['k'],
                                        sample_weight=repr_uncertainties)[0]

                    kdtree = KDTree(representations)
                    nearest_to_centroids = kdtree.query(centroids)[1]

                    query = train_dl.selection_indices[sorted_lc_indices[nearest_to_centroids]]

                all_indices.append(query)
                train_dl.select_indices(query)

    contents = {
        'training_progresses': training_progresses,
        'dev_performances': dev_performances,
        'test_performances': test_performances,
        'all_confidences': all_confidences,
        'all_indices': all_indices
    }
    contents_file_name = experiment_dir + '/contents_' + config['mode'] + '_' + \
                         str(config['train_subset_seed']) + '.pickle'
    with open(contents_file_name, 'wb') as f:
        pickle.dump(contents, f)

    # return (0, 0), None, None
    return (test_score, test_loss), None, None


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































