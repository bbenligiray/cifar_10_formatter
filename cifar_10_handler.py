import os
import cPickle

import numpy as np


def get_cifar_10():
    # downloads and extracts cifar-10 if needed
    if not os.path.isdir('cifar-10-batches-py'):
        dataset_link = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        os.system('wget -t0 -c ' + dataset_link)
        os.system('tar -xvzf cifar-10-python.tar.gz')


def read_cifar_10():
    get_cifar_10()
    # read train files
    train_files = ['data_batch_1', 'data_batch_2', 'data_batch_3',
                    'data_batch_4', 'data_batch_5']
    train_data = []
    train_labels = []
    for train_file in train_files:
        with open(os.path.join('cifar-10-batches-py', train_file), 'rb') as f:
            pickle_dict = cPickle.load(f)
            train_data.append(pickle_dict['data'])
            train_labels.append(pickle_dict['labels'])

    # concatenate train batches
    for ind_batch in range(1, len(train_data)):
        train_data[0] = np.concatenate((train_data[0], train_data[ind_batch]))
    train_data = train_data[0]
    for ind_batch in range(1, len(train_labels)):
        train_labels[0] = np.concatenate((train_labels[0], train_labels[ind_batch]))
    train_labels = train_labels[0]

    # read test files
    with open(os.path.join('cifar-10-batches-py', 'test_batch'), 'rb') as f:
        pickle_dict = cPickle.load(f)
        test_data = pickle_dict['data']
        test_labels = pickle_dict['labels']

    # read label names
    with open(os.path.join('cifar-10-batches-py', 'batches.meta'), 'rb') as f:
        pickle_dict = cPickle.load(f)
        label_names = pickle_dict['label_names']

    # reshape images
    train_data = np.reshape(train_data, (train_data.shape[0], 3, 32, 32))
    train_data = train_data.transpose([0, 2, 3, 1])
    train_data = train_data.astype(np.float32)
    test_data = np.reshape(test_data, (test_data.shape[0], 3, 32, 32))
    test_data = test_data.transpose([0, 2, 3, 1])
    test_data = test_data.astype(np.float32)

    # convert labels to one-hot
    train_labels_one_hot = np.zeros((len(train_labels), len(label_names)), dtype=np.float32)
    for ind, label in enumerate(train_labels):
        train_labels_one_hot[ind, label] = 1
    test_labels_one_hot = np.zeros((len(test_labels), len(label_names)), dtype=np.float32)
    for ind, label in enumerate(test_labels):
        test_labels_one_hot[ind, label] = 1

    return {'train_data': train_data, 'train_labels': train_labels_one_hot,
            'test_data': test_data, 'test_labels': test_labels_one_hot,
            'label_names': label_names}
