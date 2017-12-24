################################################
#                                              #
#           Implementation of Section 5        #
#           "Build the motion manifold"        #
#                                              #
################################################

import numpy as np
from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D,Dropout
from keras.models import Model,optimizers


#load data from cmu dataset
DataSet = ".\\data\\data_cmu2.npz"
preprocess = np.load('ZMUV_locomotion.npz')
Data = np.load(DataSet)['clips']

np.random.shuffle(Data)

# make data Zero Mean - Unit Variance
Xmean = preprocess['mean']
Xstd = preprocess['std']
Data = Data-Xmean/Xstd

# split train and validation data, since it's an autoencoder input and output are the same
train_length = int(np.floor(len(Data)*0.8))
train_X = test_X = Data[:train_length]
train_Y = test_Y = Data[train_length:]

#train auto-encoder
input_X = Input(shape=(240,70))
Conv_layer1 = Conv1D(256, (25,), activation='relu', padding='same',use_bias=True)(input_X)
encoded = MaxPooling1D(2,padding='same')(Conv_layer1)
Dropout_Conv = Dropout(rate=0.25,input_shape=(120,256))(Conv_layer1)
decoded = UpSampling1D(size=2)(encoded)
Dropout_Conv = Dropout(rate=0.25,input_shape=(120,256))(decoded)
Conv_layer2 = Conv1D(70,(25,), activation='relu', padding='same',use_bias=True)(Dropout_Conv)

autoencoder = Model(input_X, Conv_layer2)
autoencoder.summary()
myadam = optimizers.adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
autoencoder.compile(optimizer=myadam, loss='mse')

autoencoder.fit(train_X, test_X,
                epochs=80,
                batch_size=1,
                validation_data=(train_Y, train_Y))

# save autoencoder for future use
model_json = autoencoder.to_json()
with open(".\\model\\autoencoder_ND.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
autoencoder.save_weights(".\\model\\autoencoder_ND.h5")
print("Saved model to disk")
