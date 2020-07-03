import pickle
import torch

import pandas as pd
from torch.utils.data import Dataset, DataLoader

__VERSION__ = '2020.01.15'
__AUTHOR__ = 'byeongal'
__CONTACT__ = 'byeongal@kookmin.ac.kr'

def help():
    print("Invincea Version {}".format(__VERSION__))
    print("Train Mode")
    print("python Train.py")
    print("--train=<train.csv> : Csv file path to train.")
    print("--model=<model_path> : Path to model to save or load (Default : model.ckpt)")
    print("--batch_size=<number_of_batch_size> : (Default : 128)")
    print("--epochs=<number_of_epochs> : (Default : 100)")

class InvinceaDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        with open(self.x[index], 'rb') as f:
            x = pickle.load(f)
            x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor([self.y[index]], dtype=torch.float32)
        return x, y

    def __len__(self):
        return len(self.x)


def get_data_loader(csv_path, batch_size = 256, shuffle = True, test_mode = False):
    df = pd.read_csv(csv_path, header=None)
    file_path_list, labels = df[0].values, df[1].values
    dataset = InvinceaDataset(file_path_list, labels)
    dataloader = DataLoader(dataset, batch_size, shuffle = shuffle)
    if test_mode:
        return dataloader, file_path_list
    else:
        return dataloader