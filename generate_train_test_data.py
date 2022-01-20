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

def generate_preprocessed_data(subject_idc):
    """
    4, 8, 12: Motor imagery: T1: left vs T2: right hand
    6, 10, 14: Motor imagery: T1: hands vs T2: feet

    label map:
    0: left hand
    1: right hand
    2: fist both hands
    3: feet
    """

    dataset_path = 'dataset/physionet.org/files/eegmmidb/1.0.0/'

    run_idc = [4, 6, 8, 10, 12, 14]  # motor imagery: hands vs feet

    delta = 1. / 160.

    whole_data = []
    whole_label = []
    for subject_idx in subject_idc:
        subject_data = []
        subject_label = []
        for run_idx in run_idc:
            fname = os.path.join(dataset_path, f"S{subject_idx:03d}", f"S{subject_idx:03d}R{run_idx:02d}.edf")
            raw = read_raw_edf(fname, preload=True, verbose=False)
            eegbci.standardize(raw)  # set channel names
            montage = make_standard_montage('standard_1005')
            raw.set_montage(montage)

            # strip channel names of "." characters
            raw.rename_channels(lambda x: x.strip('.'))

            if run_idx in [4, 8, 12]:
                events, event_id = events_from_annotations(raw, event_id=dict(T1=0, T2=1), verbose=False)
            elif run_idx in [6, 10, 14]:
                events, event_id = events_from_annotations(raw, event_id=dict(T1=2, T2=3), verbose=False)

            picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

            epochs = Epochs(raw, events, event_id=event_id, tmin=0, tmax=100/160 - delta, baseline=None, preload=True, proj=False, picks=picks, verbose=False)
            
            data = epochs.get_data()
            label = epochs.events[:, -1]

            subject_data.append(data)
            subject_label.append(label)

        subject_data = np.vstack(subject_data)
        subject_label = np.hstack(subject_label)
        
        whole_data.append(subject_data)
        whole_label.append(subject_label)

    whole_data = np.vstack(whole_data).astype(np.float32)
    whole_label = np.hstack(whole_label)

    scaler = Scaler(scalings='mean')
    whole_data = scaler.fit_transform(whole_data)
    
    return whole_data, whole_label

def split_subjects(n_splits):
    subject_idc = [i for i in range(1, 110)]
    exclusions = [88, 89, 92, 100]
    subject_idc = [i for i in subject_idc if i not in exclusions]

    for i in tqdm(range(n_splits)):
        test_idc = random.sample(subject_idc, 10)
        train_idc = [i for i in subject_idc if i not in test_idc]

        idc_dict = {'train_idc': train_idc, 'test_idc': test_idc}
        json.dump(idc_dict, open(f"./dataset/train/cross_subject_data_{i}_new.json", "w"))

        X_train, y_train = generate_preprocessed_data(train_idc)
        X_test, y_test = generate_preprocessed_data(test_idc)

        cross_subject_data = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}

        pickle.dump(cross_subject_data, open(f"./dataset/train/cross_subject_data_{i}_new.pickle", "wb"))

def split_samples(n_splits):
    subject_idc = [i for i in range(1, 110)]
    exclusions = [88, 89, 92, 100]
    subject_idc = [i for i in subject_idc if i not in exclusions]
    print('Number of subjects to use:', len(subject_idc))


    print('Loading data...')
    X, y = generate_preprocessed_data(subject_idc)
    print('Load complete.')

    print('Creating train/test datasets...')
    for i in tqdm(range(n_splits)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=PARAMS['TEST_SIZE'], shuffle=True, random_state=PARAMS['RANDOM_SEEDS'][i])
        cross_subject_data = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}
        pickle.dump(cross_subject_data, open(f"./dataset/train/cross_subject_data_{i}_new.pickle", "wb"))
        idc_dict = {'train_idc': subject_idc, 'test_idc': subject_idc}
        json.dump(idc_dict, open(f"./dataset/train/cross_subject_data_{i}_new.json", "w"))
    print('DONE.')

def split_samples_20_subjects(n_splits):
    subject_idc = [i for i in range(1, 110)]
    exclusions = [88, 89, 92, 100]
    subject_idc = [i for i in subject_idc if i not in exclusions]
    subject_idc = random.sample(subject_idc, 20)
    print('Number of subjects to use:', len(subject_idc))

    print('Loading data...')
    X, y = generate_preprocessed_data(subject_idc)
    print('Load complete.')

    print('Creating train/test datasets...')
    for i in tqdm(range(n_splits)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=PARAMS['TEST_SIZE'], shuffle=True, random_state=PARAMS['RANDOM_SEEDS'][i])
        cross_subject_data = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}
        pickle.dump(cross_subject_data, open(f"./dataset/train/cross_subject_data_{i}_new_20_subjects.pickle", "wb"))
        idc_dict = {'train_idc': subject_idc, 'test_idc': subject_idc}
        json.dump(idc_dict, open(f"./dataset/train/cross_subject_data_{i}_new_20_subjects.json", "w"))
    print('DONE.')

def main():
    n_splits = 10
    split_samples(n_splits)
    split_samples_20_subjects(n_splits)
if __name__ == '__main__':
    main()