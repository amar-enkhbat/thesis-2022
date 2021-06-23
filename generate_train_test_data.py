import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import pickle
from tqdm import tqdm

random_seed = 1
random.seed(random_seed)
np.random.seed(random_seed)

def create_cross_subject_data(train_idc, test_idc):
    X_train = np.empty((0, 64))
    y_train = np.empty((0,))
    X_test = np.empty((0, 64))
    y_test = np.empty((0,))

    for subject_id in train_idc:
        df = pd.read_csv(f"dataset/physionet.org_csv_full_imagine/{subject_id}_imagine.csv")
        X_train = np.vstack((X_train, df.iloc[:, 3:].values))
        y_train = np.hstack((y_train, df["label"].values))

    for subject_id in test_idc:
        df = pd.read_csv(f"dataset/physionet.org_csv_full_imagine/{subject_id}_imagine.csv")
        X_test = np.vstack((X_test, df.iloc[:, 3:].values))
        y_test = np.hstack((y_test, df["label"].values))

    cross_subject_data = {"train_x": X_train, "train_y": y_train, "test_x": X_test, "test_y": y_test}
    return cross_subject_data


if __name__ == "__main__":
    exclusions = ["S088", "S089", "S092", "S100"]
    subjects_idc = [f"S{i:03d}" for i in range(1, 110)]
    subjects_idc = [i for i in subjects_idc if i not in exclusions]

    if not os.path.isdir("./dataset/train"):
        os.mkdir("./dataset/train")

    for i in tqdm(range(9)):
        test_idc = random.sample(subjects_idc, 10)
        train_idc = [i for i in subjects_idc if i not in test_idc]
        cross_subject_data = create_cross_subject_data(train_idc, test_idc)
        pickle.dump(cross_subject_data, open(f"./dataset/train/cross_subject_data_{i}.pickle", "wb"))
