import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr
from transformers import get_scheduler


def tolist(obj):
    if type(obj) == list:
        return obj
    return obj.tolist()


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
        steps_per_epoch = len(train_dl)
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
        self.best_model_epoch = 0
        self.best_model = None

        # Debug function to execute after each step.
        self.debug_step_function = None

    # Override
    def predict_batch(self, batch):
        print('Method predict_batch should be overridden.')
        pass

    def predict(self, dl, disable_progress_bar=False):
        pred = []
        gold = []
        for batch in tqdm(dl, disable=disable_progress_bar):
            batch_pred, batch_gold = self.predict_batch(batch)
            pred += tolist(batch_pred)
            gold += tolist(batch_gold)
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

    def train(self, disable_progress_bar, eval_zero_shot=False, early_stopping=True):
        for epoch in range(-1, self.num_epochs):

            # Train epoch.
            if epoch >= 0:
                # Model in mode train.
                self.model.train()

                # Prepare variables for evaluating model during training.
                train_loss = 0
                train_pred = []
                train_gold = []

                for batch in tqdm(self.train_dl, disable=disable_progress_bar):
                    # Zero gradients.
                    self.optimizer.zero_grad()

                    # Forward pass.
                    batch_pred, batch_gold = self.predict_batch(batch)

                    # Remember predictions for evaluation.
                    train_pred += tolist(batch_pred)
                    train_gold += tolist(batch_gold)

                    # Compute the loss. Turn gold values into tensors before.
                    if type(batch_gold) == list:
                        batch_gold = torch.FloatTensor(batch_gold)
                    loss = self.loss_function(batch_pred, batch_gold.to(self.model.device))

                    # Store loss for evaluation. Storing sum instead of mean.
                    train_loss += loss.item() * batch_gold.shape[0]

                    # Backward pass.
                    loss.backward()

                    # Step optimizer, lr_scheduler and debug_function.
                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    if self.debug_step_function is not None:
                        self.debug_step_function()

                # Print evaluation.
                train_performance = self.performance(train_pred, train_gold)
                print('Train epoch {:<4d}: loss: {:<8.4f}, score: {:<8.4f}'.format(
                    epoch + 1, train_loss / len(batch_pred), train_performance))

            if epoch >= 0 or eval_zero_shot:
                self.model.eval()
                dev_performance, dev_loss = self.score(self.dev_dl,
                                                       disable_progress_bar=disable_progress_bar)
                ast = ''
                if dev_performance > self.best_dev_performance:
                    self.best_model = self.model.state_dict()
                    self.best_dev_performance = dev_performance
                    self.best_model_epoch = epoch
                    ast = '*'
                print('Dev epoch {:<4d}: loss: {:<8.4f}, score: {:<8.4f}'.format(
                    epoch + 1, dev_loss, dev_performance) + ast)

                if early_stopping and epoch + 1 >= (self.num_epochs + 1) // 2 and \
                        self.best_model_epoch <= epoch - 2:
                    print('Early stopping')
                    break

        self.model.load_state_dict(self.best_model)

        test_performance, test_loss = self.score(
            self.test_dl, disable_progress_bar=disable_progress_bar)
        print('Test loss: {:.4f}, score: {:.4f}'.format(test_loss, test_performance))

        return test_performance, test_loss
