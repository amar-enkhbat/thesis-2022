import random
import torch

random.seed(13123123)

N_RUNS = 5
RANDOM_SEEDS = random.sample(range(1, 1000000), N_RUNS)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PARAMS = {
    'DEVICE': DEVICE,
    'N_SUBJECTS': 5,
    'N_CLASSES': 4,
    'N_CHANNELS': 64,
    'N_RUNS': N_RUNS,
    'RANDOM_SEEDS': RANDOM_SEEDS,
    'N_EPOCHS': 2,
    'BATCH_SIZE': 32,
    'SEQ_LEN': 100,
    'FCN_HIDDEN_SIZES': [256, 512, 256],
    'CNN_HIDDEN_SIZES': [16, 512],
    'CNN_N_KERNELS': 16,
    'CNN_HIDDEN_SIZE': 512,
    'CNN_KERNEL_SIZE': [64, 64],
    'RNN_HIDDEN_SIZE': 156,
    'RNN_N_LAYERS': 2,
    'TEST_SIZE': 0.2
}