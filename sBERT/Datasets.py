from datasets import load_dataset


def load_sts():
    return load_dataset("stsb_multi_mt", name="en")


def load_paws_wiki(
        datasets_path='/home/cs-folq1/rds/rds-t2-cspp025-5bF3aEHVmLU/cs-folq1/datasets/'):
    path = datasets_path + 'paws/paws-wiki/'
    train = load_dataset('csv', data_files=path + 'train.tsv', delimiter='\t')['train']
    dev = load_dataset('csv', data_files=path + 'dev.tsv', delimiter='\t')['train']
    test = load_dataset('csv', data_files=path + 'test.tsv', delimiter='\t')['train']
    return {'train': train, 'dev': dev, 'test': test}

