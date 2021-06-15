from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
from scipy.stats import spearmanr


class CrossEncoder(nn.Module):

    def __init__(self, hidden_layer_size=200, model_name='bert-base-uncased', device='cuda'):
        super(CrossEncoder, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        self.hidden_layer = nn.Linear(self.bert.config.hidden_size, hidden_layer_size)
        self.gelu = nn.GELU()
        self.output_layer = nn.Linear(hidden_layer_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.device = device
        self.to(device)

    def forward(self, **x):
        x = self.bert(**x).last_hidden_state  # size = batch x longest_length x emb_size
        x = x[:, 0, :]  # take the output of [CLS]
        x = self.gelu(self.hidden_layer(x))
        x = self.sigmoid(self.output_layer(x))
        return x

    def prepare_batch(self, batch):
        joint_sentences = [s1 + ' [SEP] ' + s2 for s1, s2 in zip(batch['sentence1'],
                                                                 batch['sentence2'])]
        return self.tokenizer(joint_sentences, padding='longest', return_tensors='pt') \
            .to(self.device)


def run_experiment_crossencoder(train_dataset, dev_dataset, test_dataset, batch_size=16,
                                num_epochs=4, lr=2e-5, device='cuda', disable_progress_bar=False):
    model = CrossEncoder(device=device)
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_function = nn.MSELoss()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    def evaluate(dataloader):
        pred = []
        gold = []
        for batch in tqdm(dataloader, disable=disable_progress_bar):
            outputs = model(**model.prepare_batch(batch)).squeeze(1)
            targets = batch['similarity_score'].float() / 5.0
            pred += outputs.detach().cpu().numpy().tolist()
            gold += targets.detach().tolist()
        return loss_function(torch.FloatTensor(pred), torch.FloatTensor(gold)).item(), \
               spearmanr(pred, gold)[0]

    for epoch in range(num_epochs):

        model.train()
        for batch in tqdm(train_dataloader, disable=disable_progress_bar):
            optimizer.zero_grad()
            outputs = model(**model.prepare_batch(batch)).squeeze(1)
            targets = batch['similarity_score'].to(device).float() / 5.0
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        print('Loss: {:.4f}, correlation: {:.4f}'.format(*evaluate(dev_dataloader)))

    test_loss, test_corr = evaluate(test_dataloader)
    print('Test loss: {:.4f}, correlation: {:.4f}'.format(test_loss, test_corr))
    return test_loss, test_corr
