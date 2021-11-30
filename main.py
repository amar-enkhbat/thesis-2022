import os
import random
import torch
import numpy as np 
from sklearn.model_selection import train_test_split
from utils import load_data, prepare_data, prepare_data_cnn, prepare_data_rnn, print_classification_report, plot_history, plot_cm, plot_adj, plot_adj_sym
from models import FCN, CNN, RNN, GCN, GCNAuto, GCNAuto_2
from params import PARAMS
from train import get_dataloaders, train_model, init_model_params

import pickle
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

def run_model(random_seed, dataset_name, model, results_path):    
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    X_train, y_train, X_test, y_test, label_map = load_data(dataset_name)
    class_names = list(label_map.keys())

    if 'cnn' in results_path:
        X_train, y_train = prepare_data_cnn(X_train, y_train, PARAMS['SEQ_LEN'])
        X_test, y_test = prepare_data_cnn(X_test, y_test, PARAMS['SEQ_LEN'])
    elif 'rnn' in results_path:
        X_train, y_train = prepare_data_rnn(X_train, y_train, PARAMS['SEQ_LEN'])
        X_test, y_test = prepare_data_rnn(X_test, y_test, PARAMS['SEQ_LEN'])
    elif 'gcn' in results_path:
        X_train, y_train = prepare_data(X_train, y_train, PARAMS['SEQ_LEN'])
        X_test, y_test = prepare_data(X_test, y_test, PARAMS['SEQ_LEN'])
    elif 'fcn' in results_path:
        X_train, y_train = prepare_data(X_train, y_train, PARAMS['SEQ_LEN'])
        X_test, y_test = prepare_data(X_test, y_test, PARAMS['SEQ_LEN'])

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=PARAMS['TEST_SIZE'], shuffle=True, random_state=random_seed, stratify=y_train)

    dataloaders, dataset_sizes = get_dataloaders(X_train, y_train, X_valid, y_valid, X_test, y_test, PARAMS['BATCH_SIZE'], random_seed=random_seed)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    best_model, history = train_model(dataloaders, dataset_sizes, model, criterion, optimizer, PARAMS['N_EPOCHS'], random_seed=random_seed)

    y_preds, y_test = model_predict(best_model, test_loader=dataloaders['test'])

    cr, cm, auroc = print_classification_report(y_test, y_preds, PARAMS['N_CLASSES'], class_names)

    plot_history(history, results_path)
    plot_cm(cm, class_names, results_path)
    if 'gcn' in results_path:
        plot_adj(model.node_embeddings.cpu().detach().numpy(), results_path)
        A = torch.mm(model.node_embeddings, model.node_embeddings.T)
        plot_adj_sym(A.cpu().detach().numpy(), results_path)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    results = {'history': history, 'cm': cm.tolist(), 'cr': cr,'auroc': auroc , 'n_params': n_params, 'class_names': class_names}
    with open(os.path.join(results_path, 'results.pickle'), 'wb') as f:
        pickle.dump(results, f)


def model_picker(model_name, device):
    if model_name == 'imagine_fcn':
        model = FCN(in_features=PARAMS['SEQ_LEN'], num_classes=PARAMS['N_CLASSES'], n_nodes=PARAMS['N_CHANNELS'], hidden_sizes=PARAMS['FCN_HIDDEN_SIZES'])
    elif model_name == 'imagine_cnn':
        model = CNN(PARAMS['CNN_KERNEL_SIZE'], PARAMS['SEQ_LEN'], PARAMS['CNN_N_KERNELS'], PARAMS['CNN_HIDDEN_SIZE'], PARAMS['N_CLASSES'])
    elif model_name == 'imagine_rnn':
        model = RNN(PARAMS['SEQ_LEN'], PARAMS['RNN_N_LAYERS'], PARAMS['RNN_HIDDEN_SIZE'], PARAMS['N_CLASSES'])
    elif model_name == 'imagine_gcn':
        model = GCN(in_features=PARAMS['SEQ_LEN'], n_nodes=PARAMS['N_CHANNELS'], num_classes=PARAMS['N_CLASSES'], hidden_sizes=PARAMS['FCN_HIDDEN_SIZES'])
    elif model_name == 'imagine_gcn_auto':
        model = GCNAuto(in_features=PARAMS['SEQ_LEN'], n_nodes=PARAMS['N_CHANNELS'], num_classes=PARAMS['N_CLASSES'], hidden_sizes=PARAMS['FCN_HIDDEN_SIZES'])
    elif model_name == 'imagine_gcn_auto_2':
        model = GCNAuto_2(in_features=PARAMS['SEQ_LEN'], n_nodes=PARAMS['N_CHANNELS'], num_classes=PARAMS['N_CLASSES'], hidden_sizes=PARAMS['FCN_HIDDEN_SIZES'])
    model = model.to(device)
    model = init_model_params(model, random_seed=random_seed)

    return model

def average(lst):
    return sum(lst) / len(lst)

def eval_runs(model_names, dataset_names, random_seeds):
    final_result = {}
    for dataset_name in dataset_names:
        for model_name in model_names:
            accs = []
            precs_macro = []
            precs_weighted = []
            recalls_macro = []
            recalls_weighted = []
            aurocs = []
            n_trainable_params = []

            for random_seed in random_seeds:
                path = os.path.join('output', model_name, dataset_name.rstrip('.pickle').lstrip('./dataset/train/'), str(random_seed), 'results.pickle')
                results = pickle.load(open(path, 'rb'))

                accs.append(results['cr']['accuracy'])
                precs_macro.append(results['cr']['macro avg']['precision'])
                precs_weighted.append(results['cr']['weighted avg']['precision'])
                recalls_macro.append(results['cr']['macro avg']['recall'])
                recalls_weighted.append(results['cr']['weighted avg']['recall'])
                aurocs.append(results['auroc'])
                n_trainable_params.append(results['n_params'])

            result = {'accuracy': [np.mean(accs), np.std(accs)], 'precision_macro': [np.mean(precs_macro), np.std(precs_macro)], 'precision_weighted': [np.mean(precs_weighted), np.std(precs_weighted)], 'recall_macro': [np.mean(recalls_macro), np.std(recalls_macro)], 'recall_weighted': [np.mean(recalls_weighted), np.std(recalls_weighted)], 'auroc': [np.mean(aurocs), np.std(aurocs)], 'n_params': average(n_trainable_params)}
            final_result[model_name] = result
    return final_result

if __name__=='__main__':
    model_names = ['imagine_fcn', 'imagine_cnn', 'imagine_rnn', 'imagine_gcn', 'imagine_gcn_auto']

    dataset_names = [f'./dataset/train/cross_subject_data_{i}_5_subjects.pickle' for i in range(2)]
    print(dataset_names)

    random_seeds = PARAMS['RANDOM_SEEDS']
    print('Random Seeds:')
    print(random_seeds)

    # For testing
    model_names = ['imagine_gcn_auto']
    random_seeds = PARAMS['RANDOM_SEEDS'][:2]

    for dataset_name in tqdm(dataset_names):
        for model_name in tqdm(model_names):
            for random_seed in tqdm(random_seeds):
                results_path = os.path.join('output', model_name, dataset_name.rstrip('.pickle').lstrip('./dataset/train/'), str(random_seed))
                os.makedirs(results_path, exist_ok=True)
                with open(os.path.join(results_path, 'params.txt'), 'w') as f:
                    f.write(str(PARAMS))
                
                model = model_picker(model_name, device=PARAMS['DEVICE'])
                run_model(random_seed=random_seed, dataset_name=dataset_name, model=model, results_path=results_path)

    final_results = eval_runs(model_names, dataset_names, random_seeds)
    for k, v in final_results.items():
        print(k)
        print(v)
