################################################
#                                              #
#     Implementation of Section 6.1 and 6.2    #
#        "Training the Feedforward Network"    #
#                                              #
################################################

import numpy as np
from keras.layers import Conv1D,MaxPooling1D,Dropout,Input,regularizers
from keras.models import Model,optimizers
from utils import Motion_Edit

#load data in the form of (None, 240, 73)
DataSet = ".\\data\\data_cmu2.npz"
Data = np.load(DataSet)['clips']

# make data Zero Mean - Unit Variance
preprocess = np.load('ZMUV_locomotion.npz')
Xmean = np.swapaxes(preprocess['Xmean'],1,2)
Xstd = np.swapaxes(preprocess['Xstd'],1,2)
Data = (Data - Xmean)/Xstd

# extract Trajectory and foot contact information / These are inputs for Our Deep net
X_decoded = Data
Upsilon_T = Data[:,:,-7:]  #[xvelocity,zvelocity,rvelocity, lh,lt,rh,rt]

# map data to motion manifold space | feature space -> hidden space
X_encoded = Motion_Edit(X_decoded)

#defining Network structure , input-shape: (None,240,7) / output-shape: (None,120,256)
input_shape = Input((240,7))
ConvLayer1 = Conv1D(filters=64,kernel_size=(45,),input_shape=(240,7),use_bias=True,activation='relu',padding='same')(input_shape)
ConvLayer2 = Conv1D(filters=128,kernel_size=(25,),use_bias=True,activation='relu',padding='same')(ConvLayer1)
ConvLayer3 = Conv1D(filters=256,kernel_size=(15,),use_bias=True,activation='relu',padding='same')(ConvLayer2)
Polling = MaxPooling1D(2,padding='same')(ConvLayer3)
Layer_Drop = Dropout(rate=0.2)(Polling)

Motion_Synthesis = Model(input_shape,Layer_Drop)

Motion_Synthesis.summary()
myadam = optimizers.adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
Motion_Synthesis.compile(optimizer=myadam,loss='mse')

#train model
Motion_Synthesis.fit(Upsilon_T,X_encoded,epochs=200,batch_size=1)

#save model for future use
model_json = Motion_Synthesis.to_json()
with open(".\\model\\motionsynthesis.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
Motion_Synthesis.save_weights(".\\model\\motionsynthesis.h5")
print("Saved model to disk")
