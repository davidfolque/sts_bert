from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn


def select_from_state_dict(state_dict, key):
    return {k.split('.', 1)[1]: v for k, v in dict(state_dict).items() if k.split('.', 1)[0] == key}


class BiEncoder(nn.Module):

    def __init__(self, mode='base', head='none', device='cuda'):
        super(BiEncoder, self).__init__()

        assert (mode in ['base-linear-pooling', 'base-mean-pooling', 'base-cls-pooling',
                         'nli-linear-pooling', 'nli-mean-pooling', 'nli-cls-pooling'])
        self.mode = mode

        assert(head in ['none', 'cos-sim', 'extra-head-sub'])
        self.head = head

        if mode == 'nli-mean-pooling':
            bert_model = 'sentence-transformers/bert-base-nli-mean-tokens'
        elif mode in ['nli-linear-pooling', 'nli-cls-pooling']:
            bert_model = 'sentence-transformers/bert-base-nli-cls-token'
        else:
            assert(mode in ['base-linear-pooling', 'base-mean-pooling', 'base-cls-pooling'])
            bert_model = 'bert-base-uncased'

        self.bert = BertModel.from_pretrained(bert_model)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)

        if self.head == 'cos-sim':
            self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        elif self.head == 'extra-head-sub':
            self.extra_head = nn.Linear(self.bert.config.hidden_size * 3, 1)
            self.sigmoid = nn.Sigmoid()
        else:
            assert(self.head == 'none')

        self.device = device
        self.to(device)

    def forward(self, **x):
        x = self.bert(**x)
        if self.mode in ['base-mean-pooling', 'nli-mean-pooling']:
            x = torch.mean(x.last_hidden_state, 1)
        elif self.mode in ['base-cls-pooling', 'nli-cls-pooling']:
            x = x.last_hidden_state[:, 0, :]
        else:
            assert(self.mode in ['base-linear-pooling', 'nli-linear-pooling'])
            x = x.pooler_output

        if self.head != 'none':
            n = x.shape[0] // 2
            if self.head == 'cos-sim':
                x = self.cos_sim(x[:n], x[n:])
            else:
                assert(self.head == 'extra-head-sub')
                x = torch.cat((x[:n], x[n:], torch.abs(x[:n] - x[n:])), dim=1)
                x = self.sigmoid(self.extra_head(x)).squeeze(1)

        return x

    def predict_batch(self, sentence1, sentence2):
        inputs = self.tokenizer(sentence1 + sentence2, padding='longest', return_tensors='pt').to(
            self.device)
        outputs = self.forward(**inputs)
        return outputs

