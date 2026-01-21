
#####
##### Import Packages
#####
import os
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
import nltk
nltk.download("all")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import time

#####
##### Import Data
#####

from preprocess import load_cached_data
data_raw = load_cached_data()

def load_data():
    text = []
    labels = []
    for i in range(len(data_raw)):
        text.append(data_raw[i]['text'])
        labels.append(data_raw[i]['label'])
    return text, labels

text, label = load_data()

# For training only use some of the data
rem = 0.8
rem = (len(text)*rem).__round__()
t = text[:rem]
l = label[:rem]

# split into training and test texts
splitat = 0.9

split = (len(t) * splitat).__round__()

train_texts = t[:split]
train_labels = l[:split]
test_texts = t[split:]
test_labels = l[split:]

#####
##### Tokenization
#####

from nltk.tokenize import word_tokenize
from collections import defaultdict

def tokenize(texts):
    max_len = 0
    tokenized_texts = []
    word2idx = {}

    # Add <pad> and <unk> tokens to the vocabulary
    word2idx['<pad>'] = 0
    word2idx['<unk>'] = 1

    # Building our vocab from the corpus starting from index 2
    idx = 2
    for sent in texts:
        tokenized_sent = word_tokenize(sent)

        # Add `tokenized_sent` to `tokenized_texts`
        tokenized_texts.append(tokenized_sent)

        # Add new token to `word2idx`
        for token in tokenized_sent:
            if token not in word2idx:
                word2idx[token] = idx
                idx += 1

        # Update `max_len`
        max_len = max(max_len, len(tokenized_sent))

    return tokenized_texts, word2idx, max_len

def encode(tokenized_texts, word2idx, max_len):
    input_ids = []
    for tokenized_sent in tokenized_texts:
        # Pad sentences to max_len
        tokenized_sent += ['<pad>'] * (max_len - len(tokenized_sent))

        # Encode tokens to input_ids
        input_id = [word2idx.get(token) for token in tokenized_sent]
        input_ids.append(input_id)

    return np.array(input_ids)

def load_pretrained_vectors(word2idx, fname):
    print("Loading pretrained vectors...")
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    # n, d = map(int, fin.readline().split())
    d=300

    # Initilize random embeddings
    embeddings = np.random.uniform(-0.25, 0.25, (len(word2idx), d))
    embeddings[word2idx['<pad>']] = np.zeros((d,))

    # Load pretrained vectors
    count = 0
    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        if word in word2idx:
            count += 1
            embeddings[word2idx[word]] = np.array(tokens[1:], dtype=np.float32)

    print(f"There are {count} / {len(word2idx)} pretrained vectors found.")

    return embeddings

print("Tokenizing...\n")
tokenized_texts_train, word2idx, max_len = tokenize(train_texts)
tokenized_texts_test, word2idx, max_len = tokenize(test_texts)
tokenized_texts, word2idx, max_len = tokenize(np.concatenate((train_texts, test_texts), axis=None))
input_ids_train = encode(tokenized_texts_train, word2idx, max_len)
input_ids_test = encode(tokenized_texts_test, word2idx, max_len)

embeddings = load_pretrained_vectors(word2idx, "cnn_models/glove.6B.300d.txt")
embeddings = torch.tensor(embeddings)

#####
##### Data Loader
#####

from torch.utils.data import (TensorDataset, DataLoader, RandomSampler, SequentialSampler)
def data_loader(train_inputs, test_inputs, train_labels, test_labels, batch_size = 50):
    # Convert data type to torch.Tensor
    train_inputs, test_inputs, train_labels, test_labels =\
    tuple(torch.tensor(data) for data in [train_inputs, test_inputs, train_labels, test_labels])

    # Specify batch_size
    batch_size = batch_size

    # Create DataLoader for training data
    train_data = TensorDataset(train_inputs, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create DataLoader for validation data
    test_data = TensorDataset(test_inputs, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return train_dataloader, test_dataloader

train_inputs = input_ids_train
test_inputs = input_ids_test
train_dataloader, test_dataloader = \
data_loader(train_inputs, test_inputs, train_labels, test_labels, batch_size=50)

#####
##### Setup Device
#####

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

#####
##### CNN Class
#####

class CNN_NLP(nn.Module):
    def __init__(self, pretrained_embedding=None,
                 freeze_embedding=False,
                 vocab_size=None,
                 embed_dim=300,
                 filter_sizes=[3, 4, 5],
                 num_filters=[100, 100, 100],
                 num_classes=2,
                 dropout=0.5):
        super(CNN_NLP, self).__init__()
        # Embedding layer
        if pretrained_embedding is not None:
            self.vocab_size, self.embed_dim = pretrained_embedding.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze_embedding)
        else:
            self.embed_dim = embed_dim
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.embed_dim, padding_idx=0, max_norm=5.0)
        # Conv Network
        self.conv1d_list = nn.ModuleList([nn.Conv1d(in_channels=self.embed_dim, out_channels=num_filters[i], kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])
        # Fully-connected layer and Dropout
        self.fc = nn.Linear(np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids):
        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
        x_embed = self.embedding(input_ids).float()

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped = x_embed.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2]) for x_conv in x_conv_list]

        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)

        # Compute logits. Output shape: (b, n_classes)
        logits = self.fc(self.dropout(x_fc))

        return logits

#####
##### Initializing model
#####

def initilize_model(pretrained_embedding=None,
                    freeze_embedding=False,
                    vocab_size=None,
                    embed_dim=300,
                    filter_sizes=[3, 4, 5],
                    num_filters=[100, 100, 100],
                    num_classes=2,
                    dropout=0.5,
                    learning_rate=0.01,
                    weight_decay=0):
    """Instantiate a CNN model and an optimizer."""

    assert (len(filter_sizes) == len(num_filters)), "filter_sizes and \
    num_filters need to be of the same length."

    # Instantiate CNN model
    cnn_model = CNN_NLP(pretrained_embedding=pretrained_embedding,
                        freeze_embedding=freeze_embedding,
                        vocab_size=vocab_size,
                        embed_dim=embed_dim,
                        filter_sizes=filter_sizes,
                        num_filters=num_filters,
                        num_classes=num_classes,
                        dropout=0.5)

    # Send model to `device` (GPU/CPU)
    cnn_model.to(device)

    # Instantiate Adadelta optimizer
    optimizer = optim.Adadelta(cnn_model.parameters(),
                               lr=learning_rate,
                               rho=0.95,
                               weight_decay=weight_decay)

    return cnn_model, optimizer

#####
##### Loss Function
#####

loss_fn = nn.CrossEntropyLoss()

#####
##### Seed for reproducibility
#####

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

#####
##### Training function
#####

def train(model, optimizer, train_dataloader, test_dataloader=None, epochs=10, model_name=""):

    # Tracking best validation accuracy
    best_accuracy = 0
    train_time = 0

    # Start training

    print("Start training...\n")
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Test Loss':^10} | {'Test Acc':^9} | {'Elapsed':^9}")
    print("-"*60)
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================

        # Tracking time and loss
        t0_epoch = time.time()
        total_loss = 0

        # Put the model into the training mode
        model.train()

        for step, batch in enumerate(train_dataloader):
            # Load batch to GPU
            b_input_ids, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids)
            # Compute loss and accumulate the loss values
            #print(b_input_ids, "\n\n")
            #print(logits, "\n\n")
            #print(b_labels, "\n\n")
            ### Error here
            loss = loss_fn(logits, b_labels)
            ###
            total_loss += loss.item()


            # Perform a backward pass to calculate gradients
            loss.backward()

            # Update parameters
            optimizer.step()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        train_time += time.time() - t0_epoch

        # =======================================
        #               Evaluation
        # =======================================
        if test_dataloader is not None:
            # After the completion of each training epoch, measure the model's
            # performance on our validation set.
            test_loss, test_accuracy = evaluate(model, test_dataloader)

            # Track the best accuracy
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(model, "./cnn_models/models/" + model_name + "_best_model.pt")

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            print(f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | {test_loss:^10.6f} | {test_accuracy:^9.2f} | {time_elapsed:^9.2f}")
                
    print("\n")
    print(f"Training complete! Best accuracy: {best_accuracy:.2f}%.")
    return best_accuracy, train_time

#####
##### Evaluation function
#####

def evaluate(model, test_dataloader):
    """After the completion of each training epoch, measure the model's
    performance on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled
    # during the test time.
    model.eval()

    # Tracking variables
    test_accuracy = []
    test_loss = []

    # For each batch in our validation set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids)

        # Compute loss
        loss = loss_fn(logits, b_labels)
        test_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        test_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    test_loss = np.mean(test_loss)
    test_accuracy = np.mean(test_accuracy)

    return test_loss, test_accuracy

# parameters count
def count_params(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Parameters: " + str(pytorch_total_params))
    print("Trainable Parameters: " + str(pytorch_trainable_params))

# cal std and mean of list
def mean_std(arr):
    arr = np.array(arr)
    mean = np.mean(arr)
    std = np.std(arr)
    return mean, std


cnn_rand, optimizer_rand = initilize_model(vocab_size=len(word2idx),
                                    embed_dim=300,
                                    learning_rate=0.5,
                                    dropout=0.5,
                                    weight_decay=1e-3)
cnn_static, optimizer_static = initilize_model(pretrained_embedding=embeddings,
                                        freeze_embedding=True,
                                        learning_rate=0.5,
                                        dropout=0.5,
                                        weight_decay=1e-3)
cnn_non_static, optimizer_non_static = initilize_model(pretrained_embedding=embeddings,
                                            freeze_embedding=False,
                                            learning_rate=0.5,
                                            dropout=0.5,
                                            weight_decay=1e-3)

# Set seed
set_seed(42)

#####
##### CNN-rand: Word vectors are randomly initialized.
#####


acc = []
train_t = []
for i in range(1):
    best_acc, train_time = train(cnn_rand, optimizer_rand, train_dataloader, test_dataloader, epochs=10, model_name="mr_cnn_rand")
    acc.append(best_acc)
    train_t.append(train_time)

# cal avg and std of acc and time
acc_mean, acc_std = mean_std(acc)
t_mean, t_std = mean_std(train_t)
print(f"Average accuracy: {acc_mean:.2f}")
print(f"Average time: {t_mean:.2f}")

#####
##### CNN-static: pretrained word vectors are used and freezed during training.
#####


acc = []
train_t = []
for i in range(1):
    best_acc, train_time = train(cnn_static, optimizer_static, train_dataloader, test_dataloader, epochs=10, model_name="mr_cnn_static")
    acc.append(best_acc)
    train_t.append(train_time)

# cal avg and std of acc and time
acc_mean, acc_std = mean_std(acc)
t_mean, t_std = mean_std(train_t)
print(f"Average accuracy: {acc_mean:.2f}")
print(f"Average time: {t_mean:.2f}")

#####
##### CNN-non-static: pretrained word vectors are fine-tuned during training.
#####

acc = []
train_t = []
for i in range(1):
    best_acc, train_time = train(cnn_non_static, optimizer_non_static, train_dataloader, test_dataloader, epochs=10, model_name="mr_cnn_non_static")
    acc.append(best_acc)
    train_t.append(train_time)

# cal avg and std of acc and time
acc_mean, acc_std = mean_std(acc)
t_mean, t_std = mean_std(train_t)
print(f"Average accuracy: {acc_mean:.2f}")
print(f"Average time: {t_mean:.2f}")


#####
##### Testing the models
#####

#cnn_non_static, optimizer = initilize_model(pretrained_embedding=embeddings, freeze_embedding=False, learning_rate=0.5, dropout=0.5, weight_decay=1e-3)
#count_params(cnn_non_static)

predict_dict = {
    0: "No Bias",
    1: "Bias"
}

def predict_review(text, model=cnn_non_static.to("cpu"), max_len=62, bias = None):

    # Tokenize, pad and encode text
    tokens = word_tokenize(text.lower())
    padded_tokens = tokens + ['<pad>'] * (max_len - len(tokens))
    input_id = [word2idx.get(token, word2idx['<unk>']) for token in padded_tokens]

    # Convert to PyTorch tensors
    input_id = torch.tensor(input_id).unsqueeze(dim=0)

    # Compute logits
    logits = model.forward(input_id)

    #  Compute probability
    probs = F.softmax(logits, dim=1).squeeze(dim=0)

    predicted_bias = predict_dict[int(probs.argmax())]
    correct = (predicted_bias == bias)
    #print(f"{probs[1] * 100:.2f}%")
    #print("Real Bias: ", bias)
    #print("Predicted Bias: ", predicted_bias)
    #print("Correct: ", correct)
    #print(probs, "\n")

    return int(probs.argmax())

test_model = [torch.load("./cnn_models/models/mr_cnn_rand_best_model.pt", weights_only=False),
            torch.load("./cnn_models/models/mr_cnn_static_best_model.pt", weights_only=False),
            torch.load("./cnn_models/models/mr_cnn_non_static_best_model.pt", weights_only=False)]
[i.to("cpu") for i in test_model]
[i.eval() for i in test_model]

predictions = [[], [], []]
for j in range(len(test_model)):
    print(f"Testing Model {j+1}...")
    for i in tqdm(range(rem, len(label))):
        prediction_text = text[i]
        bias = label[i]
        predictions[j].append(predict_review(prediction_text, model = test_model[j], bias = bias))

#####
##### Display test results
#####

x_real = np.linspace(rem, len(label)-1, len(label)-rem)
x_predict = np.linspace(rem, len(label)-1, len(label)-rem)
y_real = []
y_predict = predictions
for i in range(rem, len(label)):
    y_real.append(label[i])

correct = [0, 0, 0]
for k in range(len(test_model)):
    for i in range(len(y_real)):
        if y_real[i] == y_predict[k][i]:
            correct[k] += 1
    print(f"\n\nModel: {k+1}")
    print(f"Precision: {correct[k] / len(y_real)*100:.2f}%")
    print(f"{'Bias':<14}|{'Real':^8}|{'Predict':^8}")
    print("-"*35)
    for i in range (2):
        print(f"{predict_dict[i]:<14}|{y_real.count(i):^8}|{y_predict[k].count(i):^8}")
    print(f"{'Total':<14}|{len(y_real):^8}|{len(y_predict[k]):^8}")

    # # fig, axs = plt.subplots(1,2, figsize=(12,4), sharey='row')
    # # ax1,ax2 = axs
    # # ax1.scatter(x_real, y_real)
    # # ax2.scatter(x_predict, y_predict[k])

    # # ax1.set_title("Real Bias")
    # # ax1.set_xlabel("Article nr")
    # # ax1.set_yticks([0, 1, 2, 3, 4])
    # # ax1.set_yticklabels(["left", "leaning-left", "center", "leaning-right", "right"])

    # # ax2.set_title("Predicted Bias")
    # # ax2.set_xlabel("Article nr")
    # # plt.savefig("graphs/bias_scatter_" + str(k+1) + "_.png")

