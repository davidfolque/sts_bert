import torch
import numpy as np
from tqdm import tqdm
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

        # Store training progress.
        self.training_progress = {'train_loss': [], 'train_performance': [], 'dev_loss': [],
                                  'dev_performance': []}

        # Debug functions to execute after each step/epoch.
        self.debug_step_function = None
        self.debug_epoch_function = None

    # Override
    def predict_batch(self, batch):
        raise NotImplementedError('Method predict_batch should be overridden.')

    def predict(self, dl, disable_progress_bar=False):
        pred = []
        gold = []
        with torch.no_grad():
            for batch in tqdm(dl, disable=disable_progress_bar):
                batch_pred, batch_gold = self.predict_batch(batch)
                pred += tolist(batch_pred)
                gold += tolist(batch_gold)
        return pred, gold

    # Override
    @staticmethod
    def performance(pred, gold):
        raise NotImplementedError('Method performance should be overridden.')

    def score(self, dl, disable_progress_bar=False):
        pred, gold = self.predict(dl, disable_progress_bar=disable_progress_bar)
        loss = self.loss_function(torch.Tensor(pred).to(self.model.device),
                                  torch.Tensor(gold).to(self.model.device))
        performance = self.performance(np.array(pred), np.array(gold))
        return performance, loss

    def train(self, disable_progress_bar, eval_zero_shot=False, early_stopping=True, verbose=True):
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
                        batch_gold = torch.FloatTensor(batch_gold).to(self.model.device)
                    loss = self.loss_function(batch_pred, batch_gold)

                    # Store loss for evaluation. Storing sum instead of mean.
                    train_loss += loss.item() * batch_gold.shape[0]

                    if self.debug_step_function is not None:
                        self.debug_step_function(middle=True)

                    # Backward pass.
                    loss.backward()

                    # Step optimizer, lr_scheduler and debug_function.
                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    if self.debug_step_function is not None:
                        self.debug_step_function(middle=False)

                # Print evaluation.
                train_performance = self.performance(np.array(train_pred), np.array(train_gold))
                if verbose:
                    print('Train epoch {:<4d}: loss: {:<8.4f}, score: {:<8.4f}'.format(
                        epoch + 1, train_loss / len(batch_pred), train_performance))
                self.training_progress['train_loss'].append(train_loss / len(batch_pred))
                self.training_progress['train_performance'].append(train_performance)

                if self.debug_epoch_function is not None:
                    self.debug_epoch_function()

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
                if verbose:
                    print('Dev epoch {:<4d}: loss: {:<8.4f}, score: {:<8.4f}'.format(
                        epoch + 1, dev_loss, dev_performance) + ast)
                self.training_progress['dev_loss'].append(dev_loss.item())
                self.training_progress['dev_performance'].append(dev_performance)

                if early_stopping and epoch + 1 >= (self.num_epochs + 1) // 2 and \
                        self.best_model_epoch <= epoch - 2:
                    if verbose:
                        print('Early stopping')
                    break

        self.model.load_state_dict(self.best_model)
