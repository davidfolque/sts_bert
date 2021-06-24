from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn


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

    def predict_batch(self, sentence1, sentence2):
        inputs = self.tokenizer(sentence1 + sentence2, padding='longest', return_tensors='pt').to(
            self.device)
        outputs = self.forward(**inputs).squeeze(1)
        N = outputs.shape[0] // 2
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        pred_scores = cos(outputs[:N], outputs[N:])
        return pred_scores

