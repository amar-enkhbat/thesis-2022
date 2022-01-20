import pickle
from utils import plot_adj
from params import PARAMS
from models import GCNAuto
from main import init_model_params
from datetime import datetime
from tqdm import tqdm
import os
from main import  prepare_datasets, show_metrics, run_model

def kernel_picker(kernel_name, random_seed, results_path, device):
    model = GCNAuto(kernel_type=kernel_name, 
    in_features=PARAMS['SEQ_LEN'], 
    n_nodes=PARAMS['N_CHANNELS'], 
    num_classes=PARAMS['N_CLASSES'], 
    hidden_sizes=PARAMS['GCNAUTO_HIDDEN_SIZES'], 
    dropout_p=PARAMS['GCNAUTO_DROPOUT_P'], 
    device=PARAMS['DEVICE'])

    model = init_model_params(model, random_seed=random_seed)
    if kernel_name in 'bc':
        model.init_adj_diag()

    pickle.dump(model.adj.cpu().detach().numpy(), open(f'{results_path}/untrained_adj.pickle', 'wb'))
    plot_adj(model.adj.cpu().detach().numpy(), f'{results_path}/untrained_adj.png')

    return model.to(device)

def main():
    kernel_names = ['a', 'b', 'c', 'd', 'e']
    dataset_names = [f'cross_subject_data_{i}_new' for i in range(5)]
    dataset_names = dataset_names[:1]
    random_seeds = [0]

    time_now = datetime.now().strftime('%Y-%m-%d-%H-%M')
    for kernel_name in tqdm(kernel_names):
        for dataset_name in dataset_names:
            for random_seed in random_seeds:
                results_path = os.path.join('output', time_now, f'gcn-{kernel_name}', dataset_name, str(random_seed))
                os.makedirs(results_path, exist_ok=True)
                with open(os.path.join('output', time_now, 'params.txt'), 'w') as f:
                    f.write(str(PARAMS))

                dataloaders = prepare_datasets(random_seed, dataset_name, results_path)

                model = kernel_picker(kernel_name, random_seed, results_path, device=PARAMS['DEVICE'])
                run_model(random_seed, dataloaders, model, results_path)

    kernel_names = ['gcn-' + i for i in kernel_names]
    final_results = show_metrics(time_now, kernel_names, dataset_names, random_seeds)

    

if __name__ == '__main__':
    main()



