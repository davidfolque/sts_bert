from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
from scipy.stats import spearmanr
import numpy as np


class BiEncoder(nn.Module):

    def __init__(self, model_name='bert-base-uncased', device='cuda'):
        super(BiEncoder, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        self.device = device
        self.to(device)

    def forward(self, **x):
        x = self.bert(**x).last_hidden_state  # size = batch x longest_length x emb_size
        x = torch.mean(x, 1)
        return x

    def prepare_batch(self, batch):
        joint_sentences = batch['sentence1'] + batch['sentence2']
        return self.tokenizer(joint_sentences, padding='longest', return_tensors='pt') \
            .to(self.device)


def run_experiment_biencoder(train_dataset, dev_dataset, test_dataset, batch_size=16, num_epochs=4,
                             lr=2e-5, device='cuda', disable_progress_bar=False):
    model = BiEncoder(device=device)
    optimizer = AdamW(model.parameters(), lr=lr)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    loss_function = nn.MSELoss()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    best_loss = np.inf
    best_model = model.state_dict()

    def evaluate(dataloader):
        pred = []
        gold = []
        for batch in tqdm(dataloader, disable=disable_progress_bar):
            outputs = model(**model.prepare_batch(batch)).squeeze(1)
            N = outputs.shape[0] // 2
            pred_scores = cos(outputs[:N], outputs[N:])
            targets = batch['similarity_score'].float() / 5.0
            pred += pred_scores.detach().cpu().numpy().tolist()
            gold += targets.detach().tolist()
        return loss_function(torch.FloatTensor(pred), torch.FloatTensor(gold)).item(), \
               spearmanr(pred, gold)[0]

    for epoch in range(num_epochs):

        model.train()
        for batch in tqdm(train_dataloader, disable=disable_progress_bar):
            optimizer.zero_grad()
            outputs = model(**model.prepare_batch(batch)).squeeze(1)
            N = outputs.shape[0] // 2
            pred_scores = cos(outputs[:N], outputs[N:])
            targets = batch['similarity_score'].to(device).float() / 5.0
            loss = loss_function(pred_scores, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        dev_loss, dev_corr = evaluate(dev_dataloader)
        print('Loss: {:.4f}, correlation: {:.4f}'.format(dev_loss, dev_corr))
        if dev_loss < best_loss:
            best_model = model.state_dict()
            best_loss = dev_loss

    model.load_state_dict(best_model)
    test_loss, test_corr = evaluate(test_dataloader)
    print('Test loss: {:.4f}, correlation: {:.4f}'.format(test_loss, test_corr))
    return test_loss, test_corr
