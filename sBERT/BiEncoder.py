from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
from scipy.stats import spearmanr
from Trainer import Trainer


def select_from_state_dict(state_dict, key):
    return {k.split('.', 1)[1]: v for k, v in dict(state_dict).items() if k.split('.', 1)[0] == key}


class BiEncoder(nn.Module):

    def __init__(self, mode='base', device='cuda'):
        super(BiEncoder, self).__init__()

        assert(mode in ['base', 'pretrained-nli'])
        self.mode = mode

        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        if mode == 'pretrained-nli':
            # Load pretrained model state_dict
            path = '/home/cs-folq1/rds/rds-t2-cspp025-5bF3aEHVmLU/cs-folq1/pretrained_models/' \
                   'bert-nli/bert-base.state_dict'
            print('Loading state dict from ' + path + '.')
            state_dict = torch.load(path, map_location=device)

            # Load pretrained bert model. Setting strict=False as we don't have position_ids. But
            # they are not needed here, we can use the default ones.
            load_result = self.bert.load_state_dict(select_from_state_dict(state_dict, 'bert'),
                                                    strict=False)
            # Assert that the only keys missing are the position ids.
            assert(load_result.missing_keys == ['embeddings.position_ids'])
            assert(load_result.unexpected_keys == [])

        self.device = device
        self.to(device)

    def forward(self, **x):
        x = self.bert(**x)
        if self.mode == 'base':
            x = torch.mean(x.last_hidden_state, 1)
        else:
            assert(self.mode == 'pretrained-nli')
            x = x.pooler_output
        return x


class BiEncoderTrainer(Trainer):

    def __init__(self, model, train_dataset, dataset, num_epochs, batch_size=16, lr=2e-5,
                 lr_scheduler='constant', warmup_percent=0.0):
        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        dev_dl = DataLoader(dataset['dev'], batch_size=batch_size)
        test_dl = DataLoader(dataset['test'], batch_size=batch_size)
        optimizer = AdamW(model.parameters(), lr=lr)
        loss_function = nn.MSELoss()
        super().__init__(model=model, train_dl=train_dl, dev_dl=dev_dl, test_dl=test_dl,
                         num_epochs=num_epochs, optimizer=optimizer, lr_scheduler=lr_scheduler,
                         warmup_percent=warmup_percent, loss_function=loss_function)

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
