import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import pickle
import json

from sklearn.preprocessing import StandardScaler
SCALER = StandardScaler()

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings('ignore', category=UndefinedMetricWarning) 

from params import PARAMS
random_seed = PARAMS['RANDOM_SEEDS'][0]
random.seed(random_seed)
np.random.seed(random_seed)

def create_cross_subject_data(train_idc, test_idc):
    X_train = np.empty((0, 64))
    y_train = np.empty((0,))
    X_test = np.empty((0, 64))
    y_test = np.empty((0,))

    for subject_id in train_idc:
        df = pd.read_csv(f"dataset/physionet.org_csv_full_imagine/{subject_id}_imagine.csv")
        values = df.iloc[:, 3:].values
        labels = df["label"].values
        # Normalize per subject NOT after concat
        values = SCALER.fit_transform(values)
        X_train = np.vstack((X_train, values))
        y_train = np.hstack((y_train, labels))


    for subject_id in test_idc:
        df = pd.read_csv(f"dataset/physionet.org_csv_full_imagine/{subject_id}_imagine.csv")
        values = df.iloc[:, 3:].values
        labels = df["label"].values
        # Normalize per subject NOT after concat
        values = SCALER.fit_transform(values)
        X_test = np.vstack((X_test, values))
        y_test = np.hstack((y_test, labels))

    cross_subject_data = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}
    return cross_subject_data

def main():
    n_splits = 10
    exclusions = ["S088", "S089", "S092", "S100"]
    subjects_idc = [f"S{i:03d}" for i in range(1, 110)]
    subjects_idc = [i for i in subjects_idc if i not in exclusions]
    print('Total number of available subjets:', len(subjects_idc))

    if not os.path.isdir("./dataset/train"):
        os.mkdir("./dataset/train")

    for i in tqdm(range(n_splits)):
        test_idc = random.sample(subjects_idc, 10)
        train_idc = [i for i in subjects_idc if i not in test_idc]
        idc_dict = {'train_idc': train_idc, 'test_idc': test_idc}
        json.dump(idc_dict, open(f"./dataset/train/cross_subject_data_{i}.json", "w"))

        cross_subject_data = create_cross_subject_data(train_idc, test_idc)
        pickle.dump(cross_subject_data, open(f"./dataset/train/cross_subject_data_{i}.pickle", "wb"))

    # Example dataset with 5 subjects.
    
    subjects_idc = [f"S{i:03d}" for i in range(1, 110)]
    subjects_idc = [i for i in subjects_idc if i not in exclusions]
    n_train_subjects = 5
    n_test_subjects = 1
    n_splits = 10
    asdf = []
    asdfasdf = []
    for i in tqdm(range(n_splits)):
        train_idc = []
        test_idc = []
        for _ in range(n_train_subjects):
            random_val = random.sample(subjects_idc, 1)[0]
            subjects_idc.remove(random_val)
            train_idc.append(random_val)
        for _ in range(n_test_subjects):
            random_val = random.sample(subjects_idc, 1)[0]
            subjects_idc.remove(random_val)
            test_idc.append(random_val)
        asdf += test_idc
        asdfasdf += train_idc
        idc_dict = {'train_idc': train_idc, 'test_idc': test_idc}
        json.dump(idc_dict, open(f"./dataset/train/cross_subject_data_{i}_5_subjects.json", "w"))
        cross_subject_data = create_cross_subject_data(train_idc, test_idc)
        pickle.dump(cross_subject_data, open(f"./dataset/train/cross_subject_data_{i}_5_subjects.pickle", "wb"))

    # Dataset for each subjects.

if __name__ == "__main__":
    main()
