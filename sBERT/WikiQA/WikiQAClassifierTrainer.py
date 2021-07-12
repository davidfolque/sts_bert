from Trainer import Trainer, tolist
from WikiQA.WikiQADataLoader import WikiQAQuestionsDataLoader
from transformers import AdamW
import torch


class WikiQAClassifierTrainer(Trainer):
    def __init__(self, model, dataset, trainset_size, trainset_seed, num_epochs, batch_size=16,
                 lr=2e-5, devset_size=None):
        train_dl = WikiQAQuestionsDataLoader(dataset['train'], batch_size=batch_size,
                                             size=trainset_size, shuffle=True, seed=trainset_seed)
        dev_dl = WikiQAQuestionsDataLoader(dataset['validation'], batch_size=batch_size,
                                           size=devset_size, shuffle=devset_size is not None,
                                           seed=trainset_seed)
        test_dl = WikiQAQuestionsDataLoader(dataset['test'], batch_size=batch_size)
        optimizer = AdamW(model.parameters(), lr=lr)

        def loss_fnc(outputs, targets):
            if type(outputs) == list:
                outputs = torch.FloatTensor(outputs).to(self.model.device)
            if type(targets) == list:
                targets = torch.BoolTensor(targets).to(self.model.device)
            outputs *= targets[:, 0]
            outputs = -torch.log(torch.clamp(outputs, min=1e-5))
            return torch.sum(outputs, dim=0)

        loss_function = loss_fnc
        super().__init__(model=model, train_dl=train_dl, dev_dl=dev_dl, test_dl=test_dl,
                         num_epochs=num_epochs, optimizer=optimizer, lr_scheduler='constant',
                         warmup_percent=0.0, loss_function=loss_function)

    def predict_batch(self, batch):
        logits = self.model.predict_batch(batch['question'], batch['answer'])
        exps = torch.exp(logits)

        all_qns = batch['question_number']
        qns = list(set(all_qns))
        all_qns_t = torch.LongTensor(all_qns).to(self.model.device)
        qns_t = torch.LongTensor(qns).to(self.model.device)
        mask = torch.ones(len(all_qns), len(qns), dtype=torch.long,
                          device=self.model.device) * qns_t == all_qns_t.unsqueeze(1)

        cum_exps = torch.sum(mask * exps.unsqueeze(1), dim=0, keepdim=True)
        cum_exps = torch.sum(mask * cum_exps, dim=1, keepdim=False)

        softmaxes = exps / cum_exps

        targets = list(zip(map(float, batch['label']), batch['question_number']))
        return softmaxes, targets

    @staticmethod
    def performance(pred, gold):
        pred = tolist(pred)
        gold = tolist(gold)
        correct = 0
        current_id = -1
        currently_correct = False
        current_best_prob = -1
        for i in range(len(pred)):
            prob = pred[i]
            label, question_id = gold[i]
            if prob > current_best_prob:
                current_best_prob = prob
                currently_correct = label == 1
            if i == len(pred) - 1 or question_id != gold[i + 1][1]:
                correct += currently_correct
                current_id += 1
                currently_correct = True
                current_best_prob = -1
        return correct / current_id
