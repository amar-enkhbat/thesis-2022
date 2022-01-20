# %%
import os
import numpy as np
import mne
import pandas as pd
import random
from tqdm import tqdm

random_seed = 1
random.seed(random_seed)
np.random.seed(random_seed)

labels_remap = {276: 'idling_eyes_open',
277: 'idling_eyes_closed',
768: 'start_of_a_trial',
769: 'imagine_left', 
770: 'imagine_right', 
771: 'imagine_foot', 
772: 'imagine_tongue', 
783: 'unknown',
1023: 'rejected_trial',
1072: 'eye_movements',
32766: 'start_of_a_new_run'}

def process_annots(annot_df, freq = 250):
    
    timestep = pd.to_timedelta(f"{1 / freq} seconds")
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
def change_labels(label_df, labels_remap):
    label_df["description"] = label_df["description"].astype(int).replace(labels_remap)
    return label_df

# %%
dataset_path = 'dataset/BCICIV_2a_gdf/'
save_path = 'dataset/BCICIV_2a_gdf_csv_full/'

if not os.path.isdir(save_path):
    os.mkdir(save_path)

subjects_idc = [f"A{i:02d}T.gdf" for i in range(1, 10)]


# %%
freq = 250
timestep = pd.to_timedelta(f"{1 / freq} seconds")

# %%
diff_df = []

# %%
for subject_id in tqdm(subjects_idc):
    run_path = os.path.join(dataset_path, subject_id)
    print(run_path)
    raw = mne.io.read_raw_gdf(run_path, verbose=False)
    raw_data = raw.to_data_frame()
    raw_data = raw_data.drop(columns=["time"])
    original_data_length = raw_data.shape[0]

    raw_labels = raw.annotations.to_data_frame()
    raw_labels = change_labels(raw_labels, labels_remap)
    raw_labels = process_annots(raw_labels)

    if raw_data.shape[0] != raw_labels.shape[0]:
        len_diff = raw_data.shape[0] - raw_labels.shape[0]
        
        unique = raw_data.iloc[raw_labels.shape[0]:].values
        unique = np.unique(unique)

        diff_df.append({"filename": run_path, "length_diff": len_diff, "diff_value_unique": unique})
    
    raw_data = pd.concat([raw_labels, raw_data], axis=1)
    raw_data = raw_data.dropna()
    raw_data = raw_data.drop(columns=['EOG-left', 'EOG-central', 'EOG-right'])
    raw_data.to_csv(os.path.join(save_path, subject_id.rstrip(".gdf") + ".csv"), index=False)
    raw_labels.to_csv(os.path.join(save_path, subject_id.rstrip(".gdf") + "_labels.csv"), index=False)

diff_df = pd.DataFrame(diff_df)
diff_df.to_csv(os.path.join(save_path, "data_label_diff.csv"))

# %%
filenames = os.listdir('dataset/BCICIV_2a_gdf_csv_full/')
if not os.path.isdir('dataset/BCICIV_2a_gdf_csv_full_imagine/'):
    os.mkdir('dataset/BCICIV_2a_gdf_csv_full_imagine/')

# %%
for filename in tqdm(filenames):
    if 'label' in filename:
        continue
    df = pd.read_csv("dataset/BCICIV_2a_gdf_csv_full/" + filename)
    df = df[df["label"].str.contains("imagine")]
    df.to_csv("dataset/BCICIV_2a_gdf_csv_full_imagine/" + filename[:4] + "_imagine.csv", index=False)

# %%
# df = pd.read_csv('dataset/BCICIV_2a_gdf_csv_full_imagine/A06T_imagine.csv')

# %%
# df.iloc[:, 2:].shape

# %%
# import pickle

# data = pickle.load(open('dataset/train/cross_subject_data_bci_2a_0_5_subjects.pickle', 'rb'))

# %%
# np.unique(data['y_train'])

# %%
# data['X_train'].shape

# %%
# data['X_train'].reshape(-1, 100, 22).shape

# %%



