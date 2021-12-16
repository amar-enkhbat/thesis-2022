import warnings
warnings.filterwarnings("ignore")

import os
import random
from datetime import datetime
import torch
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import load_data, prepare_data, prepare_data_cnn, prepare_data_rnn, print_classification_report, plot_history, plot_cm, plot_adj, plot_adj_sym
from models import FCN, CNN, RNN, GCN, GCNAuto, GCRAMAuto, GATAuto, GCRAM
from params import PARAMS
from train import get_dataloaders, train_model, init_model_params

import pickle
import json
from tqdm import tqdm

def model_predict(model, test_loader):
    model.eval()
    with torch.no_grad():
        y_preds = []
        y_true = []
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            y_preds.append(preds)
            y_true.append(labels)

    y_preds = torch.cat(y_preds).tolist()
    y_true = torch.cat(y_true).tolist()


    return y_preds, y_true

def prepare_datasets(random_seed, dataset_name, results_path):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    dataset_path = os.path.join('./dataset/train', dataset_name + '.pickle')
    X_train, y_train, X_test, y_test, label_map = load_data(dataset_path)
    class_names = list(label_map.keys())

    X_train = np.vstack([X_train, X_test])
    y_train = np.hstack([y_train, y_test])

    if 'cnn' in results_path:
        X_train, y_train = prepare_data_cnn(X_train, y_train, PARAMS['SEQ_LEN'])
        # X_test, y_test = prepare_data_cnn(X_test, y_test, PARAMS['SEQ_LEN'])
    elif 'rnn' in results_path:
        X_train, y_train = prepare_data_rnn(X_train, y_train, PARAMS['SEQ_LEN'])
        # X_test, y_test = prepare_data_rnn(X_test, y_test, PARAMS['SEQ_LEN'])
    elif 'gcn' in results_path:
        X_train, y_train = prepare_data(X_train, y_train, PARAMS['SEQ_LEN'])
        # X_test, y_test = prepare_data(X_test, y_test, PARAMS['SEQ_LEN'])
    elif 'fcn' in results_path:
        X_train, y_train = prepare_data(X_train, y_train, PARAMS['SEQ_LEN'])
        # X_test, y_test = prepare_data(X_test, y_test, PARAMS['SEQ_LEN'])
    elif 'gcram' in results_path:
        X_train, y_train = prepare_data(X_train, y_train, PARAMS['SEQ_LEN'])
        # X_test, y_test = prepare_data(X_test, y_test, PARAMS['SEQ_LEN'])
    elif 'gat' in results_path:
        X_train, y_train = prepare_data(X_train, y_train, PARAMS['SEQ_LEN'])
        # X_test, y_test = prepare_data(X_test, y_test, PARAMS['SEQ_LEN'])

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=PARAMS['TEST_SIZE'], shuffle=True, random_state=random_seed)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=PARAMS['VALID_SIZE'], shuffle=True, random_state=random_seed)
    
    dataloaders, dataset_sizes = get_dataloaders(X_train, y_train, X_valid, y_valid, X_test, y_test, PARAMS['BATCH_SIZE'], random_seed=random_seed)

    return dataloaders, dataset_sizes, class_names

def run_model(random_seed, dataloaders, dataset_sizes, class_names, model, results_path):    
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=PARAMS['LR'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, PARAMS['SCHEDULER_STEPSIZE'], PARAMS['SCHEDULER_GAMMA'])

    best_model, history = train_model(dataloaders, dataset_sizes, model, criterion, optimizer, scheduler, PARAMS['N_EPOCHS'], random_seed=random_seed)

    y_preds, y_test = model_predict(best_model, test_loader=dataloaders['test'])

    cr, cm, auroc = print_classification_report(y_test, y_preds, PARAMS['N_CLASSES'], class_names)

    plot_history(history, results_path)
    plot_cm(cm, class_names, results_path)
    if 'gcn' in results_path:
        plot_adj(model.node_embeddings.cpu().detach().numpy(), results_path)
        pickle.dump(model.node_embeddings.cpu().detach().numpy(), open('./output/node_embedding.pickle', 'wb'))
        A = torch.mm(model.node_embeddings, model.node_embeddings.T)
        plot_adj_sym(A.cpu().detach().numpy(), results_path)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    results = {'history': history, 'cm': cm.tolist(), 'cr': cr,'auroc': auroc , 'n_params': n_params, 'class_names': class_names}
    with open(os.path.join(results_path, 'results.pickle'), 'wb') as f:
        pickle.dump(results, f)


def model_picker(model_name, random_seed, device):
    if model_name == 'imagine_fcn':
        model = FCN(in_features=PARAMS['SEQ_LEN'], num_classes=PARAMS['N_CLASSES'], n_nodes=PARAMS['N_CHANNELS'], hidden_sizes=PARAMS['FCN_HIDDEN_SIZES'])
    elif model_name == 'imagine_cnn':
        model = CNN(PARAMS['CNN_KERNEL_SIZE'], PARAMS['SEQ_LEN'], PARAMS['CNN_N_KERNELS'], PARAMS['CNN_HIDDEN_SIZE'], PARAMS['N_CLASSES'])
    elif model_name == 'imagine_rnn':
        model = RNN(PARAMS['SEQ_LEN'], PARAMS['RNN_N_LAYERS'], PARAMS['RNN_HIDDEN_SIZE'], PARAMS['N_CLASSES'])
    elif model_name == 'imagine_gcn':
        model = GCN(in_features=PARAMS['SEQ_LEN'], n_nodes=PARAMS['N_CHANNELS'], num_classes=PARAMS['N_CLASSES'], hidden_sizes=PARAMS['GCN_HIDDEN_SIZES'])
    elif model_name == 'imagine_gcn_auto':
        model = GCNAuto(in_features=PARAMS['SEQ_LEN'], n_nodes=PARAMS['N_CHANNELS'], num_classes=PARAMS['N_CLASSES'], hidden_sizes=PARAMS['GCNAUTO_HIDDEN_SIZES'])
    elif model_name == 'imagine_gcram_auto':
        model = GCRAMAuto(in_features=PARAMS['SEQ_LEN'], n_nodes=PARAMS['N_CHANNELS'], num_classes=PARAMS['N_CLASSES'], hidden_sizes=PARAMS['FCN_HIDDEN_SIZES'])
    elif model_name == 'imagine_gat_auto':
        model = GATAuto(in_features=PARAMS['SEQ_LEN'], n_nodes=PARAMS['N_CHANNELS'], num_classes=PARAMS['N_CLASSES'], hidden_sizes=PARAMS['FCN_HIDDEN_SIZES'])
    elif model_name == 'imagine_gcram':
        model = GCRAM(seq_len=PARAMS['SEQ_LEN'], cnn_in_channels=PARAMS['GCRAM_CNN_IN_CHANNELS'], cnn_n_kernels=PARAMS['GCRAM_CNN_N_KERNELS'], cnn_kernel_size=PARAMS['GCRAM_CNN_KERNEL_SIZE'], cnn_stride=PARAMS['GCRAM_CNN_STRIDE'], maxpool_kernel_size=PARAMS['GCRAM_MAXPOOL_KERNEL_SIZE'], maxpool_stride=PARAMS['GCRAM_MAXPOOL_STRIDE'], lstm_hidden_size=PARAMS['GCRAM_LSTM_HIDDEN_SIZE'], is_bidirectional=PARAMS['GCRAM_LSTM_IS_BIDIRECTIONAL'], lstm_n_layers=PARAMS['GCRAM_LSTM_N_LAYERS'], attn_embed_dim=PARAMS['GCRAM_ATTN_EMBED_DIM'], n_classes=PARAMS['N_CLASSES']).to(PARAMS['DEVICE'])

    model = model.to(device)
    model = init_model_params(model, random_seed=random_seed)

    return model

def show_metrics(time_now, model_names, dataset_names, random_seeds):
    final_results = []
    for model_name in model_names:
        for dataset_name in dataset_names:
            subject_idc = json.load(open(os.path.join('./dataset/train', dataset_name + '.json'), 'r'))
            accs = []
            precs_macro = []
            precs_weighted = []
            recalls_macro = []
            recalls_weighted = []
            aurocs = []
            n_trainable_params = []

            for random_seed in random_seeds:
                path = os.path.join('output', time_now, model_name, dataset_name.rstrip('.pickle').lstrip('./dataset/train/'), str(random_seed), 'results.pickle')
                results = pickle.load(open(path, 'rb'))

                accs.append(results['cr']['accuracy'])
                precs_macro.append(results['cr']['macro avg']['precision'])
                precs_weighted.append(results['cr']['weighted avg']['precision'])
                recalls_macro.append(results['cr']['macro avg']['recall'])
                recalls_weighted.append(results['cr']['weighted avg']['recall'])
                aurocs.append(results['auroc'])
                n_trainable_params.append(results['n_params'])
            result_df = pd.DataFrame({'model_name': [model_name for i in random_seeds], 'dataset_name': [dataset_name for i in random_seeds],'train_idc': [subject_idc['train_idc'] for i in random_seeds], 'test_idc': [subject_idc['test_idc'] for i in random_seeds], 'random_seed': random_seeds, 'accuracy': accs, 'precision_macro': precs_macro, 'precision_weighted': precs_weighted, 'recall_macro': recalls_macro, 'recall_weighted': recalls_weighted, 'AUROC': aurocs, 'n_params': n_trainable_params})
            final_results.append(result_df)
    final_results = pd.concat(final_results).reset_index(drop=True)
    final_results.to_csv(os.path.join('./output', time_now, 'results.csv'), index=False)

    std_per_dataset = final_results.groupby(['model_name', 'dataset_name']).std().drop(columns=['random_seed'])
    std_per_dataset = std_per_dataset.rename(columns={'accuracy': 'accuracy_std',
                        'precision_macro': 'precision_macro_std', 
                        'precision_weighted': 'precision_weighted_std', 
                        'recall_macro': 'recall_macro_std', 
                        'recall_weighted': 'recall_weighted_std'})
    std_per_dataset = std_per_dataset.drop(columns=['AUROC', 'n_params'])
    mean_per_dataset = final_results.groupby(['model_name', 'dataset_name']).mean().drop(columns='random_seed')
    results_per_dataset = pd.concat([mean_per_dataset, std_per_dataset], axis=1)
    results_per_dataset.to_csv(os.path.join('./output', time_now, 'results_per_dataset.csv'))

    std_per_model = final_results.groupby(['model_name']).std().drop(columns=['random_seed'])
    std_per_model = std_per_model.rename(columns={'accuracy': 'accuracy_std',
                        'precision_macro': 'precision_macro_std', 
                        'precision_weighted': 'precision_weighted_std', 
                        'recall_macro': 'recall_macro_std', 
                        'recall_weighted': 'recall_weighted_std'})
    std_per_model = std_per_model.drop(columns=['AUROC', 'n_params'])
    mean_per_model = final_results.groupby(['model_name']).mean().drop(columns='random_seed')
    results_per_model = pd.concat([mean_per_model, std_per_model], axis=1)
    results_per_model.to_csv(os.path.join('./output', time_now, 'results_per_model.csv'))
    print(results_per_model)
    return final_results


def main():
    model_names = ['imagine_fcn', 'imagine_cnn', 'imagine_rnn', 'imagine_gcn', 'imagine_gcn_auto', 'imagine_gcram_auto', 'imagine_gat_auto', 'imagine_gcram']

    dataset_names = [f'cross_subject_data_{i}_5_subjects' for i in range(5)]

    random_seeds = PARAMS['RANDOM_SEEDS']
    
    ### For testing ###
    # dataset_names = [f'cross_subject_data_{i}_5_subjects' for i in range(5)]
    # model_names = ['imagine_gcram']
    # model_names = ['imagine_fcn', 'imagine_cnn', 'imagine_rnn', 'imagine_gcn', 'imagine_gcn_auto', 'imagine_gcram_auto', 'imagine_gat_auto', 'imagine_gcram']
    # random_seeds = random_seeds[:1]
    # dataset_names = dataset_names[:1]
    ###################

    print('#' * 50)
    print('Model names:', model_names)
    print('Number of models:', len(model_names))
    print('Dataset names:', dataset_names)
    print('Number of datasets:', len(dataset_names))
    print('Random seeds:', random_seeds)
    print('Number of random seeds:', len(random_seeds))
    print('#' * 50)
    print('PARAMS:')
    print(PARAMS)
    print('')
    
    input_key = input('Execute Y/N?  ')

    if input_key == 'N' or input_key == 'n':
        exit(1)

    time_now = datetime.now().strftime('%Y-%m-%d-%H-%M')
    for model_name in tqdm(model_names):
        for dataset_name in dataset_names:
            for random_seed in random_seeds:
                results_path = os.path.join('output', time_now, model_name, dataset_name, str(random_seed))
                os.makedirs(results_path, exist_ok=True)
                with open(os.path.join('output', time_now, 'params.txt'), 'w') as f:
                    f.write(str(PARAMS))

                dataloaders, dataset_sizes, class_names = prepare_datasets(random_seed, dataset_name, results_path)

                
                model = model_picker(model_name, random_seed, device=PARAMS['DEVICE'])
                # summary(model, input_size=(32, 64, 100))
                run_model(random_seed, dataloaders, dataset_sizes, class_names, model, results_path)

    final_results = show_metrics(time_now, model_names, dataset_names, random_seeds)

if __name__=='__main__':
    main()