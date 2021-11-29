import os
import random
import torch
import numpy as np 
from sklearn.model_selection import train_test_split
from torch._C import device
from utils import load_data, prepare_data, prepare_data_cnn, prepare_data_rnn, print_classification_report, plot_history, plot_cm, plot_adj
from models import FCN, CNN, RNN, GCN, GCNAuto
from params import PARAMS
from train import get_dataloaders, train_model, init_model_params
from sklearn.metrics import accuracy_score

import pickle
from tqdm import tqdm

def model_predict(model, test_loader, dataset_size):
    model.eval()
    with torch.no_grad():
        y_preds = []
        y_true = []
        running_corrects = 0.0
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            y_preds.append(preds)
            y_true.append(labels)

    y_preds = torch.cat(y_preds).tolist()
    y_true = torch.cat(y_true).tolist()


    return y_preds, y_true

def run_model(random_seed, model, results_path):    
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    run_number = '5_subjects'
    X, y, label_map = load_data(f'./dataset/train/cross_subject_data_{run_number}.pickle')
    class_names = list(label_map.keys())

    if 'cnn' in results_path:
        X, y = prepare_data_cnn(X, y, PARAMS['SEQ_LEN'])
    elif 'rnn' in results_path:
        X, y = prepare_data_rnn(X, y, PARAMS['SEQ_LEN'])
    elif 'gcn' in results_path:
        X, y = prepare_data(X, y, PARAMS['SEQ_LEN'])
    elif 'fcn' in results_path:
        X, y = prepare_data(X, y, PARAMS['SEQ_LEN'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=PARAMS['TEST_SIZE'], random_state=random_seed, stratify=y)

    dataloaders, dataset_sizes = get_dataloaders(X_train, y_train, X_test, y_test, PARAMS['BATCH_SIZE'], random_seed=random_seed)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    best_model, history = train_model(dataloaders, dataset_sizes, model, criterion, optimizer, PARAMS['N_EPOCHS'], random_seed=random_seed)

    y_preds, y_test = model_predict(best_model, test_loader=dataloaders['val'], dataset_size=dataset_sizes['val'])

    cr, cm, auroc = print_classification_report(y_test, y_preds, PARAMS['N_CLASSES'], class_names)

    plot_history(history, results_path)
    plot_cm(cm, class_names, results_path)
    if 'gcn' in results_path:
        plot_adj(model.node_embeddings.cpu().detach().numpy(), results_path)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    results = {'history': history, 'cm': cm.tolist(), 'cr': cr,'auroc': auroc , 'n_params': n_params, 'class_names': class_names}
    with open(os.path.join(results_path, 'results.pickle'), 'wb') as f:
        pickle.dump(results, f)


def model_picker(model_name, device):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    model = model.to(device)
    model = init_model_params(model, random_seed=random_seed)

    return model

def average(lst):
    return sum(lst) / len(lst)

def eval_runs(model_names, random_seeds):
    final_result = {}
    for model_name in model_names:
        accs = []
        precs_macro = []
        precs_weighted = []
        recalls_macro = []
        recalls_weighted = []
        aurocs = []
        n_trainable_params = []

        for random_seed in random_seeds:
            path = os.path.join('./output', model_name, str(random_seed), 'results.pickle')
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
    model_name = model_names[0]

    random_seeds = PARAMS['RANDOM_SEEDS']
    print('Random Seeds:')
    print(random_seeds)
    for model_name in tqdm(model_names):
        # print('#' * 20)
        # print(model_name.upper())
        # print('#' * 20)

        for random_seed in random_seeds:
            results_path = os.path.join('output', model_name, str(random_seed))
            # if not os.path.isdir('output'):
            #     os.mkdir('output')
            # if not os.path.isdir(os.path.join('output', model_name)):
            #     os.mkdir(os.path.join('output', model_name))
            # if not os.path.isdir(results_path):
            #     os.mkdir(results_path)
            os.makedirs(results_path, exist_ok=True)
            with open(os.path.join(results_path, 'params.txt'), 'w') as f:
                f.write(str(PARAMS))
            
            
            model = model_picker(model_name, device=device)
            run_model(random_seed, model, results_path)

    final_results = eval_runs(model_names, random_seeds)
    for k, v in final_results.items():
        print(k)
        print(v)