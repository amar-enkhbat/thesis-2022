import random
import torch

random.seed(13123123)

N_RUNS = 10
RANDOM_SEEDS = random.sample(range(1, 1000000), N_RUNS)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PARAMS = {
    # PhysioNet
    'DEVICE': DEVICE,
    'N_CLASSES': 4,
    'N_CHANNELS': 64,
    'N_RUNS': N_RUNS,
    'RANDOM_SEEDS': RANDOM_SEEDS,
    'LABEL_MAP': {'imagine_left_fist': 0, 'imagine_right_fist': 1, 'imagine_both_fist': 2, 'imagine_both_feet': 3},

    # Global Hyperparameters
    'N_EPOCHS': 300,    
    'LR': 0.001,
    'BATCH_SIZE': 32,
    'SEQ_LEN': 100,
    'SCHEDULER_STEP_SIZE': 10,
    'SCHEDULER_GAMMA': 0.9,
    'TEST_SIZE': 1/10,
    'VALID_SIZE': 1/9,
    
    # FCN hyperparameters
    'FCN_HIDDEN_SIZES': (512, 1024, 512),
    'FCN_DROPOUT_P': 0.4,

    # CNN hyperparameters
    'CNN_HIDDEN_SIZES': (40, 512),
    'CNN_N_KERNELS': 40,
    'CNN_HIDDEN_SIZE': 512,
    'CNN_KERNEL_SIZE': (64, 45),
    'CNN_DROPOUT_P': 0.4,

    # RNN hyperparameters
    'RNN_HIDDEN_SIZE': 256,
    'RNN_N_LAYERS': 2,
    'RNN_DROPOUT_P': 0.4,

    # GCN hyperparameters
    'GCN_HIDDEN_SIZES': (512, 1024, 512),
    'GCN_DROPOUT_P': 0.4,

    # GCNAuto hyperparameters
    'GCNAUTO_KERNEL_TYPE': 'b',
    'GCNAUTO_HIDDEN_SIZES': (512, 1024, 512),
    'GCNAUTO_DROPOUT_P': 0.4,

    # GCRAM hyperparameters
    'GCRAM_GRAPH_TYPE': 'n',
    'GCRAM_CNN_IN_CHANNELS': 1,
    'GCRAM_CNN_N_KERNELS': 40,
    'GCRAM_CNN_KERNEL_SIZE': (64, 45),
    'GCRAM_CNN_STRIDE': 1,
    'GCRAM_DROPOUT1_P': 0.4,
    'GCRAM_LSTM_HIDDEN_SIZE': 64,
    'GCRAM_LSTM_IS_BIDIRECTIONAL': True,
    'GCRAM_LSTM_N_LAYERS': 2,
    'GCRAM_LSTM_DROPOUT_P': 0.4,
    'GCRAM_ATTN_EMBED_DIM': 512,
    'GCRAM_DROPOUT2_P': 0.4,
    'GCRAM_HIDDEN_SIZE': 512,

    # GCRAMAuto hyperparameters
    'GCRAMAUTO_GCN_HIDDEN_SIZE': 256,
    'GCRAMAUTO_CNN_IN_CHANNELS': 1,
    'GCRAMAUTO_CNN_N_KERNELS': 40,
    'GCRAMAUTO_CNN_KERNEL_SIZE': (64, 45),
    'GCRAMAUTO_CNN_STRIDE': 10,
    'GCRAMAUTO_DROPOUT1_P': 0.4,
    'GCRAMAUTO_LSTM_HIDDEN_SIZE': 64,
    'GCRAMAUTO_LSTM_IS_BIDIRECTIONAL': True,
    'GCRAMAUTO_LSTM_N_LAYERS': 2,
    'GCRAMAUTO_LSTM_DROPOUT_P': 0.4,
    'GCRAMAUTO_ATTN_EMBED_DIM': 512,
    'GCRAMAUTO_DROPOUT2_P': 0.4,
    'GCRAMAUTO_HIDDEN_SIZE': 512,

    ## TODO
    # GAT hyperparameters
    
    
}   