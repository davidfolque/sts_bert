import torch
import numpy as np
import gc
from sklearn.cluster import k_means
from scipy.spatial import KDTree

from ActiveLearning.ALDataLoader import ALDataLoader


class ActiveLearningRun:
    def __init__(self, train_ds, dev_ds, test_ds):
        self.train_ds = train_ds
        self.dev_ds = dev_ds
        self.test_ds = test_ds

        self.training_progresses = []
        self.dev_scores = []
        self.test_scores = []
        self.all_probs = []
        self.all_indices = []

    def create_model(self):
        raise NotImplementedError('ActiveLearningRun::create_model')

    def create_trainer(self, model, train_dl, dev_dl, test_dl, num_epochs, lr):
        raise NotImplementedError('ActiveLearningRun::create_trainer')

    def tokenize_batch(self, model, batch):
        raise NotImplementedError('ActiveLearningRun::tokenize_batch')

    def run(self, mode, initial_k, inc_k, times, n_epochs, batch_size, lr, train_subset_seed,
            min_positives=None):
        train_dl = ALDataLoader(self.train_ds, batch_size=batch_size,
                                shuffle_train=True, seed=train_subset_seed)
        if min_positives is not None:
            print('Min number of positives: {}'.format(min_positives))
        resample = True
        while resample:
            print('Sampling initial {}'.format(initial_k))
            train_dl.selected.fill(False)
            train_dl.select_k_at_random(initial_k)
            positives = np.sum([elem['label'] for i, elem in enumerate(self.train_ds) 
                                if train_dl.selected[i]])
            print('Number of positives found: {}'.format(positives))
            resample = False
            if min_positives is not None:
                resample = positives < min_positives

        # train_dl.select_k_at_random(initial_k // 2, ensure=0)
        # train_dl.select_k_at_random(initial_k // 2, ensure=1)

        dev_dl = torch.utils.data.DataLoader(self.dev_ds, batch_size=batch_size, shuffle=False)
        test_dl = torch.utils.data.DataLoader(self.test_ds, batch_size=batch_size, shuffle=False)

        self.all_indices.append(np.nonzero(train_dl.selected)[0])

        for i in range(times + 1):
            train_dl.train()

            if 'model' in dir():
                del model
            gc.collect()
            torch.cuda.empty_cache()
            model = self.create_model()

            # if original_model is not None:
            #     model.load_state_dict(original_model.state_dict())

            trainer = self.create_trainer(model=model, train_dl=train_dl, dev_dl=dev_dl,
                                          test_dl=test_dl, num_epochs=n_epochs, lr=lr)
            trainer.train(disable_progress_bar=True, verbose=False, eval_zero_shot=True,
                          early_stopping=False)
            self.training_progresses.append(trainer.training_progress)
            self.dev_scores.append(trainer.best_dev_performance)
            test_score, test_loss = trainer.score(trainer.test_dl, disable_progress_bar=False)
            self.test_scores.append(test_score)
            print('Dataloader size: {}, dev score: {:.4f}, test score: {:.4f}'.format(
                np.sum(train_dl.selected), trainer.best_dev_performance, test_score))

            if i < times:
                train_dl.selection()
                probs, _ = trainer.predict(train_dl, disable_progress_bar=False)
                probs = np.expand_dims(np.array(probs), 1)
                self.all_probs.append(probs)

                if mode == 'rnd':
                    old_selected = train_dl.selected.copy()
                    train_dl.select_k_at_random(inc_k)
                    self.all_indices.append(np.nonzero(train_dl.selected & ~old_selected)[0])
                else:
                    confidences = 2 * (np.concatenate((probs, 1 - probs), axis=1).max(axis=1) - 0.5)
                    if mode in ['mc', 'lc']:
                        if mode == 'mc':
                            confidences = 1 - confidences
                        else:
                            assert mode == 'lc'
                        lc_indices = np.argpartition(confidences, range(inc_k))[:inc_k]
                        query = train_dl.selection_indices[lc_indices]
                    else:
                        assert mode in ['lc_kmeans-mean', 'lc_kmeans-cls']
                        repr_mode = mode.split('-')[1]

                        beta = 10

                        # confidences: confidences of the unlabelled data.
                        #
                        # lc_indices: k*beta least confidences indices, sorted by confidence and
                        # pointing to the unlabelled data.
                        #
                        # representations: representations of the elements of lc_indices,
                        # but sorted by original order (all training data)
                        #
                        # sorted_lc_indices: lc_indices sorted by unlabelled data order => sorted
                        # by all training data order.

                        kbeta = inc_k * beta
                        lc_indices = np.argpartition(confidences, range(kbeta))[:kbeta]
                        sorted_lc_indices = np.sort(lc_indices)
                        temp_dl = ALDataLoader(self.train_ds, batch_size=batch_size,
                                               shuffle_train=False)
                        temp_dl.select_indices(train_dl.selection_indices[lc_indices])
                        representations = []
                        with torch.no_grad():
                            for batch in temp_dl:
                                inputs = self.tokenize_batch(model, batch)
                                outputs = model.pretrained_cross_encoder.forward_intermediate(
                                    mode=repr_mode, **inputs)
                                representations.append(outputs.cpu().numpy())
                        representations = np.concatenate(representations, axis=0)
                        repr_uncertainties = 1 - confidences[sorted_lc_indices]

                        centroids = k_means(representations, n_clusters=inc_k,
                                            sample_weight=repr_uncertainties)[0]

                        kdtree = KDTree(representations)
                        nearest_to_centroids = kdtree.query(centroids)[1]

                        query = train_dl.selection_indices[sorted_lc_indices[nearest_to_centroids]]

                    self.all_indices.append(query)
                    train_dl.select_indices(query)

