from Trainer import Trainer
from torch.utils.data import DataLoader
from transformers import AdamW
import torch.nn as nn


class PawsTrainer(Trainer):

    def __init__(self, model, dataset, num_epochs, batch_size=16, lr=2e-5):
        train_dl = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
        dev_dl = DataLoader(dataset['dev'], batch_size=batch_size)
        test_dl = DataLoader(dataset['test'], batch_size=batch_size)
        optimizer = AdamW(model.parameters(), lr=lr)
        loss_function = nn.BCELoss()
        super().__init__(model=model, train_dl=train_dl, dev_dl=dev_dl, test_dl=test_dl,
                         num_epochs=num_epochs, optimizer=optimizer, lr_scheduler='constant',
                         warmup_percent=0.0, loss_function=loss_function)

    def predict_batch(self, batch):
        outputs = self.model.predict_batch(batch['sentence1'], batch['sentence2'])
        targets = batch['label']
        return outputs, targets

    @staticmethod
    def performance(pred, gold):
        return ((pred >= 0.5) == (gold == 1)).mean()  # Accuracy.