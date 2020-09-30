import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import config
import numpy as np

def loss_fn(outputs, labels):
    #reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.view(-1)

    #mask out 'PAD' tokens
    mask = (labels > 0).float()

    #the number of tokens is the sum of elements in mask
    num_tokens = int(torch.sum(mask).item())
    #pick the values corresponding to labels and multiply by mask

    outputs = outputs[range(outputs.shape[0]), labels]*mask

    #cross entropy loss for all non 'PAD' tokens
    return -torch.sum(outputs)/num_tokens

def train_fn(data_loader, valid_loader, model, optimizer, device):

    for e in range(config.EPOCHS):
        counter = 0
        for inputs, labels in data_loader:
            model.train()
            counter += 1
            inputs, labels = inputs.to(device), labels.to(device)
            model.zero_grad()
            inputs = inputs.long()
            labels = labels.long()
            output= model(inputs)

            loss = loss_fn(output, labels)
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(model.parameters(), config.CLIP)
            optimizer.step()

            # loss stats
            if counter % config.PRINT_EVERY == 0:
                # Get validation loss
                val_losses = []
                model.eval()
                for val_inputs, val_labels in valid_loader:
                    val_inputs, val_labels = val_inputs.to(config.DEVICE), val_labels.to(config.DEVICE)
                    val_inputs = val_inputs.long()
                    val_labels = val_labels.long()
                    output = model(val_inputs)
                    output = output.to(config.DEVICE)
                    # output = model(inputs)
                    val_loss = loss_fn(output, val_labels)
                    val_losses.append(val_loss.item())

                model.train()
                print("Epoch: {}/{}...".format(e+1, config.EPOCHS),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))
    return model


def test_fn(test_loader, model, criterion, device):
    # Get test data loss and accuracy
    test_losses = []  # track loss
    num_correct = 0
    model.eval()
    # iterate over test data
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        inputs = inputs.long()
        labels = labels.long()
        # get predicted outputs
        output = model(inputs)
        # calculate loss
        test_loss = loss_fn(output, labels)
        test_losses.append(test_loss.item())

        # convert output probabilities to predicted class (0 or 1)
        pred = torch.round(output.squeeze())  # rounds to the nearest integer

        # compare predictions to true label
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy()) #if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)

    # -- stats! -- ##
    # avg test loss
    print("Test loss: {:.3f}".format(np.mean(test_losses)))

    # accuracy over all test data
    test_acc = num_correct / len(test_loader.dataset)
    print("Test accuracy: {:.3f}".format(test_acc))


def predict(model, data_object, input):

    padded_input = data_object.process_text(input)
    padded_input = torch.from_numpy(padded_input)

    # padded_input = data_object.padded_sentence[0]
    # padded_input = torch.from_numpy(padded_input)
    padded_input = padded_input.reshape(-1, 50)
    padded_input = padded_input.long()
    padded_input = padded_input.to(config.DEVICE)
    output = model(padded_input)
    ind = torch.max(output, dim=1).indices.detach().cpu().numpy()
    tags = " ".join(data_object.int_to_tag[x] for x in ind)

    output_sentence = []
    for w, i in zip(padded_input.cpu().detach().numpy()[0], ind):
        if w != 0:
            output_sentence.append((data_object.int_to_vocab[w] + '(' + data_object.int_to_tag[i]) + ')')

    sent = " ".join(x for x in output_sentence)

    return sent
