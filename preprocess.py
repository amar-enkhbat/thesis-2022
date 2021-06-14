# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import mne
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

random_seed = 1
random.seed(random_seed)
np.random.seed(random_seed)


# %%
# For remapping original labels in to interpretable labels
labels_remap = {"R01": {"T0": "eyes_open"},
          "R02": {"T0": "eyes_closed"},
          "R03": {"T0": "rest", "T1": "real_left_fist", "T2": "real_right_fist"},
          "R04": {"T0": "rest", "T1": "imagine_left_fist", "T2": "imagine_right_fist"},
          "R05": {"T0": "rest", "T1": "real_both_fist", "T2": "real_both_feet"},
          "R06": {"T0": "rest", "T1": "imagine_both_fist", "T2": "imagine_both_feet"},
          "R07": {"T0": "rest", "T1": "real_left_fist", "T2": "real_right_fist"},
          "R08": {"T0": "rest", "T1": "imagine_left_fist", "T2": "imagine_right_fist"},
          "R09": {"T0": "rest", "T1": "real_both_fist", "T2": "real_both_feet"},
          "R10": {"T0": "rest", "T1": "imagine_both_fist", "T2": "imagine_both_feet"},
          "R11": {"T0": "rest", "T1": "real_left_fist", "T2": "real_right_fist"},
          "R12": {"T0": "rest", "T1": "imagine_left_fist", "T2": "imagine_right_fist"},
          "R13": {"T0": "rest", "T1": "real_both_fist", "T2": "real_both_feet"},
          "R14": {"T0": "rest", "T1": "imagine_both_fist", "T2": "imagine_both_feet"}
         }


# %%
dataset_path = "dataset/physionet.org/files/eegmmidb/1.0.0/"
save_path = "dataset/physionet.org_csv"

if not os.path.isdir(save_path):
    os.mkdir(save_path)
    
subjects_idc = [f"S{i:03d}" for i in range(1, 110)]
# Recorded in 160 times per second
freq = 160
timestep = pd.to_timedelta(f"{1 / freq} seconds")


# %%
# Convert length annoations into timestep annotations
def process_annots(annot_df):
    new_annots = {"timestamp": [], "label": []}
    for onset, duration, description in annot_df.values:
        duration = pd.to_timedelta(f"{duration} seconds")
        stop_onset = onset + duration

        while onset != stop_onset:
            new_annots["timestamp"].append(onset)
            new_annots["label"].append(description)
            onset = onset + timestep

    new_annots = pd.DataFrame(new_annots)
    return new_annots

# Change original labels into interpretable labels
def change_labels(filename, label_df):
    filename = filename.rstrip(".edf")[-3:]
    label_df["description"] = label_df["description"].replace(labels_remap[filename])
    return label_df

# %% [markdown]
# # Convert .edf to .csv

# %%
# For recording length differences between data_df and labels_df
diff_df = []

# Convert dataset into .csv and save
for subject_id in tqdm(subjects_idc):
    runs = os.listdir(os.path.join(dataset_path, subject_id))
    runs = [i for i in runs if i.endswith(".edf")]
    runs.sort()

    for run in runs:
        run_path = os.path.join(dataset_path, subject_id, run)

        raw = mne.io.read_raw_edf(run_path, verbose=False)
        raw_data = raw.to_data_frame()
        raw_data = raw_data.drop(columns=["time"])
        original_data_length = raw_data.shape[0]

        raw_labels = raw.annotations.to_data_frame()
        raw_labels = change_labels(run, raw_labels)
        raw_labels = process_annots(raw_labels)
        

        if raw_data.shape[0] != raw_labels.shape[0]:
            len_diff = raw_data.shape[0] - raw_labels.shape[0]
            
            unique = raw_data.iloc[raw_labels.shape[0]:].values
            unique = np.unique(unique)

            diff_df.append({"filename": run, "length_diff": len_diff, "diff_value_unique": unique})
        
        raw_data = pd.concat([raw_labels, raw_data], axis=1)
        raw_data = raw_data.dropna()

        assert raw_data.shape[0] == raw_labels.shape[0] or raw_data.shape[0] == original_data_length

        if not os.path.exists(os.path.join(save_path, subject_id)):
            os.mkdir(os.path.join(save_path, subject_id))
        # raw_data.to_csv(os.path.join(save_path, subject_id, run.rstrip(".edf") + ".csv"), index=False)
        # raw_labels.to_csv(os.path.join(save_path, subject_id, run.rstrip(".edf") + "_labels.csv"), index=False)

diff_df = pd.DataFrame(diff_df)
# diff_df.to_csv(os.path.join(save_path, "data_label_diff.csv"))

# Note 1: subject 100 showed errors. 
# we're ignoring subjects:
# #88, 89, 92 100 anyway.

# Note 2: when len(raw_data.shape[0]) != len(raw_labels.shape[0]) the leftover dataframe is [0.] or [], when diff is positive or negative respectively
# #88, 92, 100 has more labels than data

# %% [markdown]
# # Concat data per subject

# %%
# Concat data of each subject

# For getting columns names only
temp = pd.read_csv("dataset/physionet.org_csv/S001/S001R01.csv")
columns = temp.columns

if not os.path.isdir("./dataset/physionet.org_csv_full/"):
    os.mkdir("./dataset/physionet.org_csv_full/")

# Save .csv files as one .csv per subject
for subject_id in tqdm(subjects_idc):
    runs = os.listdir(os.path.join(save_path, subject_id))
    runs = [i for i in runs if len(i) == 11]
    runs.sort()
    full_data = pd.DataFrame([], columns=columns)
    for run in runs:
        run_path = os.path.join(save_path, subject_id, run)
        data = pd.read_csv(run_path)
        full_data = pd.concat([full_data, data], axis=0)
    full_data = full_data.reset_index()
    full_data = full_data.rename(columns = {'index':'original_index'})
    full_data.to_csv("./dataset/physionet.org_csv_full/" + subject_id + ".csv", index=False)

# %% [markdown]
# # Extract imagined data from full

# %%
filenames = os.listdir("./dataset/physionet.org_csv_full/")
if not os.path.isdir("./dataset/physionet.org_csv_full_imagine/"):
    os.mkdir("./dataset/physionet.org_csv_full_imagine/")


# %%
for filename in tqdm(filenames):
    df = pd.read_csv("./dataset/physionet.org_csv_full/" + filename)
    df = df[df["label"].str.contains("imagine")]
    df.to_csv("./dataset/physionet.org_csv_full_imagine/" + filename[:4] + "_imagine.csv", index=False)


# %%



