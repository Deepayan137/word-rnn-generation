import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import pdb

# RNN model

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout, n_layers=1):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.fc_in = nn.Linear(input_size,hidden_size)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.lstm_1 = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        encoded = self.drop(self.encoder(x))
        output, hidden = self.lstm_1(encoded)
        output = self.drop(output)
        decoded = self.decoder((output[:,-1,:]))
        return decoded
    def init_hidden(self):
    	return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))

class deep_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout, n_layers=1):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.fc_in = nn.Linear(input_size,hidden_size)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.lstm_1 = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.lstm_2 = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        encoded = self.drop(self.encoder(x))
        output, hidden = self.lstm_1(encoded)
        output, hidden = self.lstm_1(output)
        output = self.drop(output)
        decoded = self.decoder((output[:,-1,:]))
        return decoded
    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))

