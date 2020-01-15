import pickle
import numpy as np
from tensorflow.keras.utils import Sequence

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

class DataSequence(Sequence):
    def __init__(self, x_set, y_set, batch_size, shuffle=True):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, fn_list):
        vector_array = []
        for fn in fn_list:
            with open(fn, 'rb') as f:
                vector_array.append(pickle.load(f))
        return np.array(vector_array)

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        batch_x = self.x[indexes]
        batch_y = self.y[indexes]

        return self.__data_generation(batch_x), batch_y