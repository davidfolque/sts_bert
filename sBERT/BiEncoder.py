from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
from scipy.stats import spearmanr
from Trainer import Trainer


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


class BiEncoderTrainer(Trainer):

    def __init__(self, model, train_dataset, dataset, batch_size=16, lr=2e-5):
        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        dev_dl = DataLoader(dataset['dev'], batch_size=batch_size)
        test_dl = DataLoader(dataset['test'], batch_size=batch_size)
        optimizer = AdamW(model.parameters(), lr=lr)
        loss_function = nn.MSELoss()
        super().__init__(model=model, train_dl=train_dl, dev_dl=dev_dl, test_dl=test_dl,
                         optimizer=optimizer, loss_function=loss_function)

    def predict_batch(self, batch):
        joint_sentences = batch['sentence1'] + batch['sentence2']
        inputs = self.model.tokenizer(joint_sentences, padding='longest', return_tensors='pt').to(
            self.model.device)
        outputs = self.model(**inputs).squeeze(1)
        N = outputs.shape[0] // 2
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        pred_scores = cos(outputs[:N], outputs[N:])
        targets = batch['similarity_score'].float() / 5.0
        return pred_scores, targets

    @staticmethod
    def performance(pred, gold):
        return spearmanr(pred, gold)[0]
