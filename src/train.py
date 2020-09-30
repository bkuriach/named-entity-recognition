
import NER.src.config as config
import NER.src.dataset as dataset
import NER.src.model as model
import NER.src.engine as engine
import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

def data_split(encoded_features, encoded_labels):
    split_idx = int(len(encoded_features) * config.SPLIT)
    train_x, remaining_x = encoded_features[:split_idx], encoded_features[split_idx:]
    train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]

    test_idx = int(len(remaining_x) * 0.5)
    val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
    val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

    ## print out the shapes of your resultant feature data
    print("\t\t\tFeature Shapes:")
    print("Train set: \t\t{}".format(train_x.shape),
          "\nValidation set: \t{}".format(val_x.shape),
          "\nTest set: \t\t{}".format(test_x.shape))

    return train_x, train_y, val_x, val_y, test_x, test_y


ner_data = dataset.NERData()
ner_data.load_data()
sentences = ner_data.sentence_getter()
vocab_to_int, tag_to_int = ner_data.vocab_dict()
ner_data.encode_text()
ner_data.pad_features()
encoded_features= ner_data.padded_sentence
encoded_labels = ner_data.padded_tag

train_x, train_y, val_x, val_y, test_x, test_y = data_split(encoded_features, encoded_labels)
# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

# make sure the SHUFFLE your training data
train_loader = DataLoader(train_data, shuffle=True, batch_size=config.BATCH_SIZE,drop_last=True)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=config.BATCH_SIZE,drop_last=True)
test_loader = DataLoader(test_data, shuffle=True, batch_size=config.BATCH_SIZE,drop_last=True)


vocab_size = len(ner_data.vocab_to_int)+1
output_size = len(ner_data.tag_to_int)+1

params = {}
params['vocab_size'] = vocab_size
params['output_size'] = output_size
params['embedding_dim'] = config.EMBEDDING_DIM
params['lstm_hidden_dim']= config.HIDDEN_DIM
params['n_layers'] = 1

net = model.Net(params)
# net = Net(params)

lr=0.001
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
net.to(device=config.DEVICE)
net.train()

net = engine.train_fn(train_loader, valid_loader, net, optimizer, config.DEVICE)

input = " Thousands of demonstrators have marched through London to protest the war in Iraq and demand the withdrawal of British troops from that country."
engine.predict(net, ner_data, input)

input = " Iraq demanded withdrawal of British troops. China also extended their support to Iran and Iraq"
engine.predict(net, ner_data, input)

input = " The Boeing Company is a great organization and David Calhoun is its CEO"
engine.predict(net, ner_data, input)


