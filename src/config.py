import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

SEQ_LENGTH = 50
SPLIT = 0.8
BATCH_SIZE = 100
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EMBEDDING_DIM = 150
HIDDEN_DIM = 256
N_LAYERS = 2
OUTPUT_SIZE = 1
EPOCHS = 4
PRINT_EVERY = 100
CLIP=5
MODEL_ARCH='LSTM'