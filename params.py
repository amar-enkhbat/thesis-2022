import random
import torch

random.seed(13123123)

N_RUNS = 5
RANDOM_SEEDS = random.sample(range(1, 1000000), N_RUNS)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PARAMS = {
    # PhysioNet
    # 'DEVICE': DEVICE,
    # 'N_CLASSES': 4,
    # 'N_CHANNELS': 64,
    # 'N_RUNS': N_RUNS,
    # 'RANDOM_SEEDS': RANDOM_SEEDS,
    # 'LABEL_MAP': {'imagine_left_fist': 0, 'imagine_right_fist': 1, 'imagine_both_feet': 2, 'imagine_both_fist': 3},

    # BCI
    'DEVICE': DEVICE,
    'N_CLASSES': 4,
    'N_CHANNELS': 22,
    'N_RUNS': N_RUNS,
    'RANDOM_SEEDS': RANDOM_SEEDS,
    'LABEL_MAP': {'imagine_left': 0, 'imagine_right': 1, 'imagine_foot': 2, 'imagine_tongue': 3},

    # Global Hyperparameters
    'N_EPOCHS': 300,
    'LR': 0.001,
    'BATCH_SIZE': 64,
    'SEQ_LEN': 100,
    'SCHEDULER_STEP_SIZE': 10,
    'SCHEDULER_GAMMA': 0.9,
    'TEST_SIZE': 1/10,
    'VALID_SIZE': 1/9,
    
    # FCN hyperparameters
    'FCN_HIDDEN_SIZES': (256, 512, 256),
    'FCN_DROPOUT_P': 0.4,

    # CNN hyperparameters
    'CNN_HIDDEN_SIZES': (16, 512),
    'CNN_N_KERNELS': 16,
    'CNN_HIDDEN_SIZE': 512,
    # 'CNN_KERNEL_SIZE': (64, 64),
    'CNN_KERNEL_SIZE': (22, 64),
    'CNN_DROPOUT_P': 0.4,

    # RNN hyperparameters
    'RNN_HIDDEN_SIZE': 256,
    'RNN_N_LAYERS': 2,
    'RNN_DROPOUT_P': 0.4,

    # GCN hyperparameters
    'GCN_HIDDEN_SIZES': (256, 512, 256),
    'GCN_DROPOUT_P': 0.4,

    # GCNAuto hyperparameters
    'GCNAUTO_KERNEL_TYPE': 'a',
    'GCNAUTO_HIDDEN_SIZES': (256, 512, 256),
    'GCNAUTO_DROPOUT_P': 0.4,

    # GCRAM hyperparameters
    'GCRAM_CNN_IN_CHANNELS': 1,
    'GCRAM_CNN_N_KERNELS': 20,
    # 'GCRAM_CNN_KERNEL_SIZE': (64, 64),
    'GCRAM_CNN_KERNEL_SIZE': (22, 64),
    'GCRAM_CNN_STRIDE': 1,
    'GCRAM_MAXPOOL_KERNEL_SIZE': (1, 20),
    'GCRAM_MAXPOOL_STRIDE': 2,
    'GCRAM_DROPOUT1_P': 0.4,
    # 'GCRAM_LSTM_HIDDEN_SIZE': 64,
    'GCRAM_LSTM_HIDDEN_SIZE': 22,
    'GCRAM_LSTM_IS_BIDIRECTIONAL': True,
    'GCRAM_LSTM_N_LAYERS': 2,
    'GCRAM_LSTM_DROPOUT_P': 0.4,
    'GCRAM_ATTN_EMBED_DIM': 512,
    'GCRAM_ATTN_N_HEADS': 2,
    'GCRAM_DROPOUT2_P': 0.4,
    'GCRAM_HIDDEN_SIZE': 512,

    # GAT hyperparameters
    
}   