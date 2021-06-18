import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr
from transformers import get_scheduler


class Trainer:
    def __init__(self, model, train_dl, dev_dl, test_dl, num_epochs, optimizer,
                 lr_scheduler='constant', warmup_percent=0.0, loss_function=None):
        self.model = model
        self.train_dl = train_dl
        self.dev_dl = dev_dl
        self.test_dl = test_dl
        self.num_epochs = num_epochs
        self.optimizer = optimizer

        # LR scheduler.
        steps_per_epoch = (len(train_dl) - 1) // train_dl.batch_size + 1
        total_training_steps = num_epochs * steps_per_epoch
        if lr_scheduler == 'constant':
            lr_scheduler = 'constant_with_warmup'
        warmup_steps = int(total_training_steps * warmup_percent)
        print('Scheduler type: {}, epochs: {}, steps per epoch: {}, total steps: {}, '
              'warmup steps: {}'.format(lr_scheduler, num_epochs, steps_per_epoch,
                                        total_training_steps, warmup_steps))
        self.lr_scheduler = get_scheduler(lr_scheduler, optimizer, num_warmup_steps=warmup_steps,
                                          num_training_steps=total_training_steps)

        self.loss_function = loss_function

        # For early stopping.
        self.best_dev_performance = -np.inf
        self.best_model = None

    # Override
    def predict_batch(self, batch):
        print('Method predict_batch should be overridden.')
        pass

    def predict(self, dl, disable_progress_bar=False):
        pred = []
        gold = []
        for batch in tqdm(dl, disable=disable_progress_bar):
            batch_pred, batch_gold = self.predict_batch(batch)
            pred += batch_pred.detach().cpu().numpy().tolist()
            gold += batch_gold.detach().tolist()
        return pred, gold

    # Override
    @staticmethod
    def performance(pred, gold):
        print('Method performance should be overridden.')
        pass

    def score(self, dl, disable_progress_bar=False):
        pred, gold = self.predict(dl, disable_progress_bar=disable_progress_bar)
        loss = self.loss_function(torch.Tensor(pred), torch.Tensor(gold))
        performance = self.performance(np.array(pred), np.array(gold))
        return performance, loss

    def train(self, disable_progress_bar):
        for epoch in range(self.num_epochs):

            self.model.train()
            for batch in tqdm(self.train_dl, disable=disable_progress_bar):
                self.optimizer.zero_grad()
                batch_pred, batch_gold = self.predict_batch(batch)
                loss = self.loss_function(batch_pred, batch_gold.to(self.model.device))
                loss.backward()
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

            self.model.eval()
            dev_performance, dev_loss = self.score(self.dev_dl,
                                                   disable_progress_bar=disable_progress_bar)
            ast = ''
            if dev_performance > self.best_dev_performance:
                self.best_model = self.model.state_dict()
                self.best_dev_performance = dev_performance
                ast = '*'
            print('Epoch {:<4d}: loss: {:<8.4f}, score: {:<8.4f}'.format(
                epoch + 1, dev_loss, dev_performance) + ast)

        self.model.load_state_dict(self.best_model)

        test_performance, test_loss = self.score(
            self.test_dl, disable_progress_bar=disable_progress_bar)
        print('Test loss: {:.4f}, score: {:.4f}'.format(test_loss, test_performance))

        return test_performance, test_loss
