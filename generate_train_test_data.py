import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import pickle

from sklearn.preprocessing import StandardScaler
SCALER = StandardScaler()

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
    exclusions = ["S088", "S089", "S092", "S100"]
    subjects_idc = [f"S{i:03d}" for i in range(1, 110)]
    subjects_idc = [i for i in subjects_idc if i not in exclusions]

    if not os.path.isdir("./dataset/train"):
        os.mkdir("./dataset/train")

    for i in tqdm(range(9)):
        test_idc = random.sample(subjects_idc, 10)
        train_idc = [i for i in subjects_idc if i not in test_idc]

        with open(f"./dataset/train/cross_subject_data_{i}.txt", "w") as text_file:
            idc_dict = {'train_idc': train_idc, 'test_idc': test_idc}
            text_file.write(f"{str(idc_dict)}")

        cross_subject_data = create_cross_subject_data(train_idc, test_idc)
        pickle.dump(cross_subject_data, open(f"./dataset/train/cross_subject_data_{i}.pickle", "wb"))

    # Example dataset with 5 subjects.
    

    for i in tqdm(range(9)):
        train_idc = random.sample(subjects_idc, 5)
        test_idc = [i for i in subjects_idc if i not in train_idc]
        test_idc = list(np.random.choice(test_idc, 1))
 
        with open(f"./dataset/train/cross_subject_data_{i}_5_subjects.txt", "w") as text_file:
            idc_dict = {'train_idc': train_idc, 'test_idc': test_idc}
            text_file.write(f"{str(idc_dict)}")

        cross_subject_data = create_cross_subject_data(train_idc, test_idc)
        pickle.dump(cross_subject_data, open(f"./dataset/train/cross_subject_data_{i}_5_subjects.pickle", "wb"))


if __name__ == "__main__":
    main()
