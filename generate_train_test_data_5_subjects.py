import os
import numpy as np
import pandas as pd
import random
import pickle

random_seed = 1
random.seed(random_seed)
np.random.seed(random_seed)

def create_cross_subject_data(train_idc):
    X_train = np.empty((0, 64))
    y_train = np.empty((0,))

    for subject_id in train_idc:
        df = pd.read_csv(f"dataset/physionet.org_csv_full_imagine/{subject_id}_imagine.csv")
        X_train = np.vstack((X_train, df.iloc[:, 3:].values))
        y_train = np.hstack((y_train, df["label"].values))

    cross_subject_data = {"train_x": X_train, "train_y": y_train}
    return cross_subject_data


if __name__ == "__main__":
    exclusions = ["S088", "S089", "S092", "S100"]
    subjects_idc = [f"S{i:03d}" for i in range(1, 110)]
    subjects_idc = [i for i in subjects_idc if i not in exclusions]
    subjects_idc = list(np.random.choice(subjects_idc, 5))

    if not os.path.isdir("./dataset/train"):
        os.mkdir("./dataset/train")

    with open("dataset/train/cross_subject_data_5_subjects_idc.txt", "w") as text_file:
        text_file.write(f"{str(subjects_idc)}")

    cross_subject_data = create_cross_subject_data(subjects_idc)
    pickle.dump(cross_subject_data, open(f"./dataset/train/cross_subject_data_5_subjects.pickle", "wb"))

    print('Save completed.')
    print('Saved subjects:', subjects_idc)
    print('Data length:', cross_subject_data['train_x'].shape)
