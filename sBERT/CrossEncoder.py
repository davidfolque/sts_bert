from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
from scipy.stats import spearmanr
import numpy as np
from Trainer import Trainer


class CrossEncoder(nn.Module):

    def __init__(self, hidden_layer_size=200, sigmoid_temperature=0.1,
                 model_name='bert-base-uncased',
                 device='cuda'):
        super(CrossEncoder, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        self.hidden_layer = nn.Linear(self.bert.config.hidden_size, hidden_layer_size)
        self.gelu = nn.GELU()
        self.output_layer = nn.Linear(hidden_layer_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.sigmoid_temperature = sigmoid_temperature
        self.device = device
        self.to(device)

    def forward(self, **x):
        x = self.bert(**x).last_hidden_state  # size = batch x longest_length x emb_size
        x = x[:, 0, :]  # take the output of [CLS]
        x = self.gelu(self.hidden_layer(x))
        x = self.sigmoid(self.sigmoid_temperature * self.output_layer(x))
        return x


class CrossEncoderTrainer(Trainer):

    def __init__(self, model, train_dataset, dataset, batch_size=16, lr=1e-5):
        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        dev_dl = DataLoader(dataset['dev'], batch_size=batch_size)
        test_dl = DataLoader(dataset['test'], batch_size=batch_size)
        optimizer = AdamW(model.parameters(), lr=lr)
        loss_function = nn.MSELoss()
        super().__init__(model=model, train_dl=train_dl, dev_dl=dev_dl, test_dl=test_dl,
                         optimizer=optimizer, loss_function=loss_function)

    def predict_batch(self, batch):
        joint_sentences = [s1 + ' [SEP] ' + s2 for s1, s2 in
                           zip(batch['sentence1'], batch['sentence2'])]
        inputs = self.model.tokenizer(joint_sentences, padding='longest', return_tensors='pt').to(
            self.model.device)
        outputs = self.model(**inputs).squeeze(1)
        targets = batch['similarity_score'].float() / 5.0
        return outputs, targets

    @staticmethod
    def performance(pred, gold):
        return spearmanr(pred, gold)[0]


