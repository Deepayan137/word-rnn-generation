import torch
from torchvision import transforms, datasets
from torch.autograd import Variable
import torch.nn as nn
#from data_loader import TextLoader, ToTensor
from model import LSTM
from random import shuffle
import numpy as np
import pdb
import argparse
from build_dataset import *
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"

argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str)
argparser.add_argument('--n_epochs', type=int, default= 2)
argparser.add_argument('--hidden_size', type=int, default=256)
argparser.add_argument('--n_layers', type=int, default=2)
argparser.add_argument('--learning_rate', type=float, default= 0.001)
argparser.add_argument('--chunk_len', type=int, default=200)
argparser.add_argument('--seq_len', type=int, default=30)
argparser.add_argument('--dropout', type=float, default=0.3)
args = argparser.parse_args()




X,y,vocab_size = build_dataset()

X,y = np.array(X), np.array(y)
train_size = len(X)
batch_size = 128
print(y.shape)
def save(model_ft):
        save_filename = args.filename
        torch.save(model_ft, save_filename)
        print('Saved as %s' % save_filename)


def train_model(model_ft, criterion, optimizer, n_epochs):

        for epoch in range(n_epochs):
                model_ft.train()
                hidden = model_ft.init_hidden()
                shuffle(X)
                correct =0
                for i in range(0, train_size, batch_size):
                        num_samples = min(batch_size, train_size - i)
                        data = torch.from_numpy(X[i:i+num_samples]).cuda()
                        target = torch.from_numpy(y[i:i+num_samples]).cuda()
                        data = Variable(data)
                        target = Variable(target)

                        optimizer.zero_grad()
                        output = model_ft(data, hidden)
                        loss = criterion(torch.squeeze(output), target)
                        loss.backward()
                        optimizer.step()
                        pred = output.data.max(1)[1]
                        correct += pred.eq(target.data).cpu().sum()
                        accuracy = float(correct)/train_size*100
                        if i == 18944:
                            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f Accuracy: %.4f' %((epoch+1), 
                                args.n_epochs, i+1, train_size, loss.data[0], accuracy))

        print("Saving..")
        save(model_ft)
        return(model_ft)

model_ft = LSTM(vocab_size, args.hidden_size, vocab_size, args.dropout, args.n_layers).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_ft.parameters(), args.learning_rate)
try:
        train_model(model_ft, criterion, optimizer, args.n_epochs)
except KeyboardInterrupt:
        print("Saving before quit...")
        save(model_ft)
