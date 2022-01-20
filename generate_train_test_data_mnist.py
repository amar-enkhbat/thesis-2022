import os
import numpy as np
from tqdm import tqdm
import random
import json
import pickle

from mne.channels import make_standard_montage
from mne.io import read_raw_edf
from mne.datasets import eegbci
from mne import events_from_annotations, Epochs, pick_types
from mne.decoding import Scaler

from sklearn.model_selection import train_test_split
from params import PARAMS

import torch, torchvision

random_seed = PARAMS['RANDOM_SEEDS'][0]
random.seed(random_seed)
np.random.seed(random_seed)

class Sqeeze(object):
        def __call__(self, sample):
            sqzd = torch.squeeze(sample)
            return sqzd
        
        def __repr__(self) -> str:
            return 'Squeze()'

def generate_preprocessed_data_mnist():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081)),
        Sqeeze()
    ])

    mnist_train = torchvision.datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)

    X_train = mnist_train.data.numpy()
    y_train = mnist_train.targets.numpy()
    X_test = mnist_test.data.numpy()
    y_test = mnist_test.targets.numpy()

    mnist_data = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}

    pickle.dump(mnist_data, open(f"./dataset/train/mnist_data.pickle", "wb"))

    idc_dict = {'train_idc': [i for i in range(10)], 'test_idc': [i for i in range(10)]}
    json.dump(idc_dict, open(f"./dataset/train/mnist_data.json", "w"))

def main():
    generate_preprocessed_data_mnist()

if __name__ == '__main__':
    main()