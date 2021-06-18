from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import get_scheduler
from tqdm import tqdm
from scipy.stats import spearmanr
import numpy as np
from Trainer import Trainer


def select_from_state_dict(state_dict, key):
    return {k.split('.', 1)[1]: v for k, v in dict(state_dict).items() if k.split('.', 1)[0] == key}


class CrossEncoder(nn.Module):

    # Modes: cls-pooling-hl, mean-pooling-hl, linear-pooling, pretrained-nli
    def __init__(self, hidden_layer_size=200, sigmoid_temperature=10, mode='cls',
                 pretrained_nli_label_num=3, device='cuda'):
        super(CrossEncoder, self).__init__()

        assert(mode in ['cls-pooling', 'mean-pooling', 'linear-pooling', 'pretrained_nli'])
        self.mode = mode

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        if self.mode in ['cls-pooling', 'mean-pooling']:
            self.hidden_layer = nn.Linear(self.bert.config.hidden_size, hidden_layer_size)
            self.gelu = nn.GELU()
            output_layer_in_size = hidden_layer_size
        elif self.mode == 'linear-pooling':
            output_layer_in_size = self.bert.config.hidden_size
        else:
            assert(self.mode == 'pretrained_nli')

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

            # Create and load the pretrained nli head.
            self.nli_head = nn.Linear(self.bert.config.hidden_size, pretrained_nli_label_num)
            self.nli_head.load_state_dict(select_from_state_dict(state_dict, 'nli_head'))
            print('Pretrained NLI model successfully loaded.')

            self.softmax = nn.Softmax(dim=1)
            output_layer_in_size = pretrained_nli_label_num

        self.output_layer = nn.Linear(output_layer_in_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.sigmoid_temperature = sigmoid_temperature
        self.device = device
        self.to(device)

    def forward(self, **x):
        x = self.bert(**x)

        if self.mode == 'cls-pooling':
            x = x.last_hidden_state[:, 0, :] # Take the output of [CLS].
            x = self.gelu(self.hidden_layer(x))
        elif self.mode == 'mean-pooling':
            x = torch.mean(x.last_hidden_state, dim=1) # Take the mean of all tokens' embeddings.
            x = self.gelu(self.hidden_layer(x))
        elif self.mode == 'linear-pooling':
            x = x.pooler_output
        else:
            assert(self.mode == 'pretrained_nli')
            x = self.softmax(self.nli_head(x.pooler_output))

        x = self.output_layer(x)
        x = self.sigmoid(x / self.sigmoid_temperature)
        return x


class CrossEncoderTrainer(Trainer):

    def __init__(self, model, train_dataset, dataset, num_epochs, batch_size=16, lr=2e-5,
                 lr_scheduler='constant', warmup_percent=0.0):
        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        dev_dl = DataLoader(dataset['dev'], batch_size=batch_size)
        test_dl = DataLoader(dataset['test'], batch_size=batch_size)
        optimizer = AdamW(model.parameters(), lr=lr)

        steps_per_epoch = (len(train_dataset) - 1) // batch_size + 1
        total_training_steps = num_epochs * steps_per_epoch
        if lr_scheduler == 'constant':
            lr_scheduler = 'constant_with_warmup'
        warmup_steps = int(total_training_steps * warmup_percent)
        print('Scheduler type: {}, epochs: {}, steps per epoch: {}, total steps: {}, '
              'warmup steps: {}'.format(lr_scheduler, num_epochs, steps_per_epoch,
                                        total_training_steps, warmup_steps))
        lr_scheduler = get_scheduler(lr_scheduler, optimizer, num_warmup_steps=warmup_steps,
                                     num_training_steps=total_training_steps)

        loss_function = nn.MSELoss()
        super().__init__(model=model, train_dl=train_dl, dev_dl=dev_dl, test_dl=test_dl,
                         optimizer=optimizer, lr_scheduler=lr_scheduler,
                         loss_function=loss_function)

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


