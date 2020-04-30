import sys
import numpy as np

sys.path.append('ingestion/')
from dataset import Dataset

def analysis(dataset):
    print(dataset.get_fea_table().describe())
    #for index, row in dataset.get_fea_table().iteritems():
    #    print(index, row.nunique())
    print(dataset.get_edge().describe())
    print(dataset.get_metadata().get('time_budget'))
    print(dataset.get_metadata().get('n_class'))

if __name__ == '__main__':
    data_name = ['a', 'b', 'c', 'd', 'e']
    for name in [data_name[4]]:
        dataset = Dataset('data/{}/train.data'.format(name))
        analysis(dataset)
