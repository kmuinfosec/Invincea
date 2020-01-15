import os
import sys
import getopt

import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import utils
import Invincea

def train(train_csv_path, model_path, batch_size, epochs):
    try:
        train_df = pd.read_csv(train_csv_path, header=None)
        train_data, train_label = train_df[0].values, train_df[1].values
    except Exception as e:
        print(e)
        sys.exit(1)
    model = Invincea.Invincea()
    if model_path == None:
        model_path = 'model.ckpt'
    if os.path.isfile(model_path):
        model.load_weights(model_path)

    ear = EarlyStopping(monitor='acc', patience=4)
    mcp = ModelCheckpoint(model_path,
                          monitor="acc",
                          save_best_only=True,
                          save_weights_only=False)
    train_generator = utils.DataSequence(train_data, train_label, batch_size, True)
    number_of_gpu = len(tf.config.experimental.list_physical_devices('GPU'))
    if number_of_gpu >= 2:
        parallel_model = multi_gpu_model(model, gpus=number_of_gpu)
        parallel_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
        try:
            parallel_model.fit_generator(
                train_generator,
                epochs=epochs,
                callbacks=[ear, mcp],
                workers=os.cpu_count(),
                use_multiprocessing=True,
                verbose=1,
            )
        except KeyboardInterrupt:
            model.save(model_path)
        model.save(model_path)
    else:
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
        model.fit_generator(
            train_generator,
            epochs=epochs,
            callbacks=[ear, mcp],
            workers=os.cpu_count(),
            use_multiprocessing=True,
            verbose=1,
        )

def main(argv):
    try:
        train_csv_path = None
        model_path = None
        batch_size = 128
        epochs = 100
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