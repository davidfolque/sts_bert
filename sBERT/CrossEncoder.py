from transformers import BertConfig, BertModel, BertTokenizer
import torch
import torch.nn as nn
import numpy as np


def select_from_state_dict(state_dict, key):
    return {k.split('.', 1)[1]: v for k, v in dict(state_dict).items() if k.split('.', 1)[0] == key}


class CrossEncoder(nn.Module):

    def __init__(self, hidden_layer_size=200, sigmoid_temperature=10, mode='cls-pooling',
                 pretrained_nli_label_num=3, device='cuda', toy_model=False):
        super(CrossEncoder, self).__init__()

        assert(mode in ['cls-pooling', 'cls-pooling-hidden', 'mean-pooling', 'mean-pooling-hidden',
                        'linear-pooling', 'nli-base', 'nli-head'])
        self.mode = mode

        if toy_model:
            bert_config = BertConfig.from_pretrained('bert-base-uncased')
            bert_config.num_hidden_layers = 1
            self.bert = BertModel(bert_config)
        else:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        if self.mode in ['cls-pooling-hidden', 'mean-pooling-hidden']:
            self.hidden_layer = nn.Linear(self.bert.config.hidden_size, hidden_layer_size)
            self.gelu = nn.GELU()
            output_layer_in_size = hidden_layer_size
        elif self.mode in ['cls-pooling', 'mean-pooling', 'linear-pooling']:
            output_layer_in_size = self.bert.config.hidden_size
        else:
            assert(self.mode in ['nli-base', 'nli-head'])

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

            if self.mode == 'nli-head':
                # Create and load the pretrained nli head.
                self.nli_head = nn.Linear(self.bert.config.hidden_size, pretrained_nli_label_num)
                self.nli_head.load_state_dict(select_from_state_dict(state_dict, 'nli_head'))
                output_layer_in_size = pretrained_nli_label_num
            else:
                output_layer_in_size = self.bert.config.hidden_size

            print('Pretrained NLI model successfully loaded.')

        self.output_layer = nn.Linear(output_layer_in_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.sigmoid_temperature = sigmoid_temperature
        self.device = device
        self.to(device)

    def forward(self, input_ids, token_type_ids, attention_mask):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                      token_type_ids=token_type_ids)

        if self.mode in ['cls-pooling', 'cls-pooling-hidden']:
            x = x.last_hidden_state[:, 0, :] # Take the output of [CLS].
        elif self.mode in ['mean-pooling', 'mean-pooling-hidden']:
            x = torch.mean(x.last_hidden_state, dim=1) # Take the mean of all tokens' embeddings.
        elif self.mode in ['linear-pooling', 'nli-base']:
            x = x.pooler_output
        else:
            assert(self.mode == 'nli-head')
            x = self.nli_head(x.pooler_output)

        if self.mode in ['cls-pooling-hidden', 'mean-pooling-hidden']:
            x = self.gelu(self.hidden_layer(x))

        x = self.output_layer(x)
        x = self.sigmoid(x / self.sigmoid_temperature)
        return x

    def predict_batch(self, sentence1, sentence2):
        N = len(sentence1) // 2
        flipped1 = sentence1[:N] + sentence2[N:]
        flipped2 = sentence2[:N] + sentence1[N:]
        inputs = self.tokenizer(flipped1, flipped2, padding='longest', return_tensors='pt').to(
            self.device)
        outputs = self.forward(**inputs).squeeze(1)
        return outputs


class CrossEncoderPretrained(nn.Module):

    def __init__(self, pretrained_cross_encoder, mode='replace-head'):
        super(CrossEncoderPretrained, self).__init__()

        self.pretrained_cross_encoder = pretrained_cross_encoder

        assert(mode in ['as-is', 'replace-head', 'shift-bias', 'additional-head'])
        self.mode = mode
        pretrained_head = pretrained_cross_encoder.output_layer
        if mode == 'replace-head':
            new_linear = nn.Linear(pretrained_head.in_features, pretrained_head.out_features)
            pretrained_head.weight = new_linear.weight
            pretrained_head.bias = new_linear.bias
        elif mode == 'shift-bias':
            # sigmoid(x) = 1/(1+exp(-x))
            # y = 1/(1+exp(-x)), exp(-x)=(1-y)/y, x=-log((1-y)/y)
            # sg(x) = 1/(1+exp(-x)) = 0.8 => x = -log(0.25)
            # sg(x+b) = 1/(1+exp(-x-b)) = 0.5 => x+b = -log(1)
            pretrained_head.bias.data += np.log(0.25)
        elif mode == 'additional-head':
            self.extra_head = nn.Linear(pretrained_head.out_features, 1)
            self.sigmoid = nn.Sigmoid()
        else:
            assert(mode == 'as-is')

        self.device = pretrained_cross_encoder.device
        self.to(self.device)

    def forward(self, input_ids, token_type_ids, attention_mask):
        x = self.pretrained_cross_encoder(input_ids=input_ids, attention_mask=attention_mask,
                                          token_type_ids=token_type_ids)

        if self.mode == 'additional-head':
            x = self.extra_head(x)
            x = self.sigmoid(x)

        return x

    def predict_batch(self, sentence1, sentence2):
        inputs = self.pretrained_cross_encoder.tokenizer(sentence1, sentence2, padding='longest',
                                                         return_tensors='pt').to(self.device)
        outputs = self.forward(**inputs).squeeze(1)
        return outputs
