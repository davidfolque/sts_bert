import numpy as np


class WikiQAQuestionsDLIterator:
    def __init__(self, dataset, questions_idx, batch_size, shuffle=False):
        self.dataset = dataset
        self.questions_idx = questions_idx

        # This permutation ensures that we always return the questions in a random order.
        if shuffle:
            np.random.shuffle(self.questions_idx)
        self.batch_size = batch_size
        self.nq = 0

    def __next__(self):
        if self.nq >= len(self.questions_idx):
            raise StopIteration

        question = []
        answer = []
        label = []
        question_number = []

        while self.nq < len(self.questions_idx) and len(question) + self.questions_idx[self.nq][
            1] <= self.batch_size:
            for i in range(self.questions_idx[self.nq][1]):
                element = self.dataset[self.questions_idx[self.nq][0] + i]
                question.append(element['question'])
                answer.append(element['answer'])
                label.append(element['label'])
                question_number.append(self.nq)
            self.nq += 1

        return {'question': question, 'answer': answer, 'label': label,
                'question_number': question_number}


class WikiQAPairsDLIterator:
    def __init__(self, dataset, correct_answers, incorrect_answers, batch_size, shuffle=False):
        self.dataset = dataset
        self.correct_answers = correct_answers
        self.incorrect_answers = incorrect_answers

        # These permutations ensure that we always return the questions in a random order.
        if shuffle:
            np.random.shuffle(self.correct_answers)
            np.random.shuffle(self.incorrect_answers)
        self.batch_size = batch_size
        self.nq = 0

    def __next__(self):
        if self.nq >= len(self.correct_answers):
            raise StopIteration

        question = []
        answer = []
        label = []
        question_number = []

        while self.nq < len(self.correct_answers) and len(question) + 2 <= self.batch_size:
            pos_element = self.dataset[self.correct_answers[self.nq]]
            neg_element = self.dataset[self.incorrect_answers[self.nq]]
            question += [pos_element['question'], neg_element['question']]
            answer += [pos_element['answer'], neg_element['answer']]
            label += [pos_element['label'], neg_element['label']]
            question_number += [self.nq, self.nq]

            self.nq += 1

        return {'question': question, 'answer': answer, 'label': label,
                'question_number': question_number}


class WikiQAAllDLIterator:
    def __init__(self, dataset, all_answers, batch_size, shuffle=False):
        self.dataset = dataset
        self.all_answers = all_answers

        # These permutations ensure that we always return the questions in a random order.
        if shuffle:
            np.random.shuffle(self.all_answers)
        self.batch_size = batch_size
        self.nq = 0

    def __next__(self):
        if self.nq >= len(self.all_answers):
            raise StopIteration

        question = []
        answer = []
        label = []
        question_number = []

        while self.nq < len(self.all_answers) and len(question) + 1 <= self.batch_size:
            element = self.dataset[self.all_answers[self.nq]]
            question += [element['question']]
            answer += [element['answer']]
            label += [element['label']]
            question_number += [self.nq]

            self.nq += 1

        return {'question': question, 'answer': answer, 'label': label,
                'question_number': question_number}


class WikiQADataLoader:
    def __init__(self, dataset, batch_size, size=None, shuffle=False, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.size = len(dataset) if size is None else size
        self.shuffle = shuffle

        all_questions_idx = []
        last_question_id = ''
        for i, element in enumerate(dataset):
            question_id = element['question_id']
            if question_id != last_question_id:
                all_questions_idx.append(i)
                last_question_id = question_id
        all_questions_len = np.diff(all_questions_idx + [len(dataset)])

        # This permutation ensures that we will select k questions randomly. It does not ensure
        # that we will always return them in a random order.
        if shuffle:
            question_perm = np.random.default_rng(seed=seed).permutation(len(all_questions_idx))
        else:
            question_perm = range(len(all_questions_idx))

        self.questions_idx = []
        current_size = 0
        warning_issued = False
        for idx in question_perm:
            if current_size + all_questions_len[idx] > self.size:
                break
            if not warning_issued and all_questions_len[idx] > self.batch_size:
                print('Warning: Question has more answers than the batch size')
                warning_issued = True
            self.questions_idx.append((all_questions_idx[idx], all_questions_len[idx]))
            current_size += all_questions_len[idx]


class WikiQAQuestionsDataLoader(WikiQADataLoader):
    def __iter__(self):
        return WikiQAQuestionsDLIterator(dataset=self.dataset, questions_idx=self.questions_idx,
                                         batch_size=self.batch_size, shuffle=self.shuffle)

    def __len__(self):
        if 'warning_issued' not in dir(self):
            print('Warning: WikiQAQuestionsDataLoader::__len__ called, returning a lower bound.')
            self.warning_issued = True
        return self.size // self.batch_size


class WikiQAPairsDataLoader(WikiQADataLoader):
    def __init__(self, dataset, batch_size, size=None, shuffle=False, seed=0):
        super().__init__(dataset=dataset, batch_size=batch_size, size=size, shuffle=shuffle,
                         seed=seed)

        self.correct_answers = []
        self.incorrect_answers = []
        for i in range(len(self.questions_idx)):
            offset, length = self.questions_idx[i]
            for j in range(offset, offset + length):
                if dataset[j]['label'] == 1:
                    self.correct_answers.append(j)
                else:
                    self.incorrect_answers.append(j)

    def __iter__(self):
        return WikiQAPairsDLIterator(dataset=self.dataset, correct_answers=self.correct_answers,
                                     incorrect_answers=self.incorrect_answers,
                                     batch_size=self.batch_size, shuffle=self.shuffle)

    def __len__(self):
        real_batch_size = self.batch_size // 2 * 2
        n_samples = 2 * len(self.correct_answers)
        return (n_samples - 1) // real_batch_size + 1


class WikiQAAllDataLoader(WikiQADataLoader):
    def __init__(self, dataset, batch_size, size=None, shuffle=False, seed=0):
        super().__init__(dataset=dataset, batch_size=batch_size, size=size, shuffle=shuffle,
                         seed=seed)

        self.all_answers = []
        for i in range(len(self.questions_idx)):
            offset, length = self.questions_idx[i]
            self.all_answers += list(range(offset, offset + length))

    def __iter__(self):
        return WikiQAAllDLIterator(dataset=self.dataset, all_answers=self.all_answers,
                                   batch_size=self.batch_size, shuffle=self.shuffle)

    def __len__(self):
        real_batch_size = self.batch_size // 2 * 2
        n_samples = len(self.all_answers)
        return (n_samples - 1) // real_batch_size + 1


