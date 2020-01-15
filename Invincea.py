from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Input

def Invincea():
    input_layer = Input((1025, ))
    x = Dense(1025, activation='relu')(input_layer)
    x = Dense(1025, activation='relu')(x)
    output_layer = Dense(1, activation='sigmoid')(x)
    return Model(input_layer, output_layer)