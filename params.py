import random
import torch

random.seed(13123123)

N_RUNS = 5
RANDOM_SEEDS = random.sample(range(1, 1000000), N_RUNS)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PARAMS = {
    # Environment
    'DEVICE': DEVICE,
    'N_SUBJECTS': 5,
    'N_CLASSES': 4,
    'N_CHANNELS': 64,
    'N_RUNS': N_RUNS,
    'RANDOM_SEEDS': RANDOM_SEEDS,

    # Global Hyperparameters
    'N_EPOCHS': 100,
    'LR': 0.001,
    'BATCH_SIZE': 32,
    'SEQ_LEN': 100,
    'SCHEDULER_STEPSIZE': 1.0,
    'SCHEDULER_GAMMA': 0.95,
    
    # FCN hyperparameters
    'FCN_HIDDEN_SIZES': (256, 512, 256),

    # CNN hyperparameters
    'CNN_HIDDEN_SIZES': (16, 512),
    'CNN_N_KERNELS': 16,
    'CNN_HIDDEN_SIZE': 512,
    'CNN_KERNEL_SIZE': (64, 64),

    # RNN hyperparameters
    'RNN_HIDDEN_SIZE': 256,
    'RNN_N_LAYERS': 2,
    'DROPOUT_P': 0.0,
    'TEST_SIZE': 1/6,
    'VALID_SIZE': 1/5,

    # GCN hyperparameters
    'GCN_HIDDEN_SIZES': (256, 512, 256),

    # GCNAuto hyperparameters
    'GCNAUTO_HIDDEN_SIZES': (256, 512, 256),

}   