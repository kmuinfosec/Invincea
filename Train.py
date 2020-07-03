import os
import sys
import getopt

import torch
import torch.optim as optim
import torch.functional as F
from tqdm import tqdm

import utils
import Invincea

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

def train_model(model, epochs, train_loader, optimizer):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0
        total = 0
        with tqdm(train_loader, desc='Train Epoch #{}'.format(epoch)) as t:
            for data, target in t:
                data, target = data.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                output = model(data)
                optimizer.zero_grad()
                loss = F.binary_cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                total += len(data)
                total_correct += output.round().eq(target).sum().item()
                total_loss += loss.item() * len(data)
                t.set_postfix(loss='{:.4f}'.format(total_loss / total), accuracy='{:.4f}'.format(total_correct / total))

def train(train_csv_path, model_path, batch_size, epochs):
    try:
        train_loader = utils.get_data_loader(train_csv_path, batch_size, True)
    except Exception as e:
        print(e)
        sys.exit(1)
    model = Invincea.Invincea().to(DEVICE)
    optimizer = optim.Adam(model.parameters())
    if model_path == None:
        model_path = 'model.dat'
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))
    train_loader(model, epochs, train_loader, optimizer)
    torch.save(model.state_dict(), model_path)

def main(argv):
    try:
        train_csv_path = None
        model_path = None
        batch_size = 128
        epochs = 10
        optlist, args = getopt.getopt(argv[1:], '', ['help', 'train=', 'model=', 'batch_size=', 'epochs=',])
        for opt, arg in optlist:
            if opt == '--help':
                utils.help()
                sys.exit(0)
            elif opt == '--train':
                train_csv_path = arg
            elif opt == '--model':
                model_path = arg
            elif opt == '--batch_size':
                batch_size = int(arg)
            elif opt == '--epochs':
                epochs = int(arg)
        if train_csv_path == None:
            print('The following values must be input')
            print('train')
            utils.help()
            sys.exit(1)
        train(train_csv_path, model_path, batch_size, epochs)
    except Exception as e:
        print(e)
        sys.exit(1)

if __name__ == '__main__':
    main(sys.argv)