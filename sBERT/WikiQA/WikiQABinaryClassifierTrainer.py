from Trainer import Trainer, tolist
from WikiQA.WikiQADataLoader import WikiQAAllDataLoader, WikiQAPairsDataLoader
from transformers import AdamW
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import pandas as pd


class WikiQABinaryClassifierTrainerBase(Trainer):
    def __init__(self, *args, **kwargs):
        self.sigmoid = torch.nn.Sigmoid()
        super().__init__(*args, **kwargs)

    def update_scaled_loss_function(self, is_scaled):
        if is_scaled:
            n_pos = 0
            n_neg = 0
            for batch in self.train_dl:
                n_pos_batch = (np.array(batch['label']) > 0.5).sum()
                n_pos += n_pos_batch
                n_neg += len(batch['label']) - n_pos_batch
            print('+: {}, -: {}'.format(n_pos, n_neg))

            def loss_function(outputs, targets):
                weights = (targets[:, 0] * n_neg + (1 - targets[:, 0]) * n_pos) / (
                            n_neg + n_pos)
                return torch.nn.BCELoss(weights)(outputs, targets[:, 0])
        else:
            def loss_function(outputs, targets):
                return torch.nn.BCELoss()(outputs, targets[:, 0])

        self.loss_function = loss_function

    def predict_batch(self, batch):
        logits = self.model.predict_batch(batch['question'], batch['answer'])
        outputs = self.sigmoid(logits)
        # Adding labels twice, as we don't need the question numbers. This could be improved, as
        # we don't need to pass 2 things anymore?
        targets = list(zip(map(float, batch['label']), batch['label']))
        return outputs, targets

    @staticmethod
    def performance(pred, gold, verbose=False):
        # F1

        if type(gold) == list:
            gold = np.array(gold)
            pred = np.array(pred)
        cm = confusion_matrix(gold[:, 0], pred > 0.5, labels=[0, 1])
        if verbose:
            print(pd.DataFrame(cm, columns=['Predicted-', 'Predicted+'], index=['Gold-', 'Gold+']))
        precision, recall, f1, support = precision_recall_fscore_support(gold[:, 0], pred > 0.5,
                                                                         average='binary')
        if verbose:
            print('Precision {:.2f}, recall {:.2f}, F1 {:.2f}'.format(precision, recall, f1))
        return f1

        # Accuracy
        # return ((pred >= 0.5) == (gold[:, 0] >= 0.5)).mean()

    # @staticmethod
    # def performance(pred, gold):
    #     pred = np.array(tolist(pred))
    #     gold = np.array(tolist(gold))
    #
    #     cm = confusion_matrix(gold[:, 0], pred > 0.5, labels=[0, 1])
    #     print(pd.DataFrame(cm, columns=['Predicted-', 'Predicted+'], index=['Gold-', 'Gold+']),
    #           flush=True)
    #
    #     question_start = 0
    #     ap_sum = 0.0
    #     num_questions = 0
    #     for i in range(len(pred)):
    #         question_id = gold[i, 1]
    #         if i == len(pred) - 1 or question_id != gold[i + 1][1]:
    #             ap_sum += average_precision_score(gold[question_start:(i + 1), 0],
    #                                               pred[question_start:(i + 1)])
    #             question_start = i + 1
    #             num_questions += 1
    #     return ap_sum / num_questions


class WikiQABinaryClassifierTrainer(WikiQABinaryClassifierTrainerBase):
    def __init__(self, model, dataset, mode, trainset_size, trainset_seed, num_epochs,
                 batch_size=16, lr=2e-5, devset_size=None):
        assert mode in ['all-unscaled', 'all-scaled', 'downsampling']
        self.mode = mode

        if self.mode == 'downsampling':
            train_dl = WikiQAPairsDataLoader(dataset['train'], batch_size=batch_size,
                                             size=trainset_size, shuffle=True, seed=trainset_seed)
        else:
            assert self.mode in ['all-unscaled', 'all-scaled']
            train_dl = WikiQAAllDataLoader(dataset['train'], batch_size=batch_size,
                                           size=trainset_size, shuffle=True, seed=trainset_seed)

        dev_dl = WikiQAAllDataLoader(dataset['validation'], batch_size=batch_size, size=devset_size,
                                     shuffle=devset_size is not None, seed=trainset_seed)
        test_dl = WikiQAAllDataLoader(dataset['test'], batch_size=batch_size)
        optimizer = AdamW(model.parameters(), lr=lr)

        super().__init__(model=model, train_dl=train_dl, dev_dl=dev_dl, test_dl=test_dl,
                         num_epochs=num_epochs, optimizer=optimizer, lr_scheduler='constant',
                         warmup_percent=0.0, loss_function=None)

        self.update_scaled_loss_function(self.mode == 'all-scaled')


class WikiQABinaryClassifierForALTrainer(WikiQABinaryClassifierTrainerBase):
    def __init__(self, model, train_dl, dev_dl, test_dl, mode, num_epochs, lr=2e-5):
        assert mode in ['scaled', 'unscaled']
        self.mode = mode

        optimizer = AdamW(model.parameters(), lr=lr)

        super().__init__(model=model, train_dl=train_dl, dev_dl=dev_dl, test_dl=test_dl,
                         num_epochs=num_epochs, optimizer=optimizer, lr_scheduler='constant',
                         warmup_percent=0.0, loss_function=None)

        assert train_dl.training
        self.update_scaled_loss_function(is_scaled=(self.mode == 'scaled'))
