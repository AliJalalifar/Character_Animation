################################################
#                                              #
#         Implementation of Section 6.3        #
#        "Disambiguation for Locomotion"       #
#                                              #
################################################

import numpy as np
from keras.layers import Input,Conv1D,Dropout
from keras.models import Model
from keras.callbacks import TensorBoard

data_address = ".\\data\data_cmu2.npz"
data_train = np.load(data_address)['clips']

X = data_train
X = np.swapaxes(X, 1, 2)
preprocess = np.load('ZMUV_locomotion.npz')
X = (X - preprocess['Xmean']) / preprocess['Xstd']

#Extracting Trajectory and foot_contact information from CMU Dataset
Trajectory = X[:, -7:-4]
Foot_contact = X[:, -4:]

# Gama is the vector : [omega,tau_lh,tau_lt,tau_rh,tau_rt]
Gama = np.zeros((len(Foot_contact), 5, 240))

# Frequency and Step Duration
for i in range(len(Foot_contact)):

    # Compute Step Duration,Tau, for every four joints : Left heel,Left toe,Right heel,Right toe
    Step_duration = np.zeros(Foot_contact[i].shape)

    for j in range(0, Foot_contact[i].shape[1]):
        window = slice(max(j - 60, 0), min(j + 60, Foot_contact[i].shape[1]))

        # compute the ratio of foot joints are up to the ratio of frames foot joint are down
        d = np.mean((Foot_contact[i, :, window] > 0), axis=1)  # Feet is down
        u = np.mean((Foot_contact[i, :, window] < 0), axis=1)  # Feet is up
        Step_duration[:, j] = ((np.pi * d) / (u + d))  # compute ratio
        Tau = np.cos(((np.pi * d) / (u + d)))  # See section 6.3

    # for every joint we should calculate omega
    Step_Frequency = np.zeros(np.shape(Foot_contact[i]))

    for j in range(np.shape(Foot_contact[i])[0]):
        toggled_frame = -1
        for k in range(1, np.shape(Foot_contact[i])[1]):
            if toggled_frame == -1 and (Foot_contact[i, j, k - 1] * Foot_contact[i, j, k]) < 0:
                toggled_frame = k
            elif (Foot_contact[i, j, k - 1] * Foot_contact[i, j, k]) < 0:
                Step_Frequency[j, toggled_frame:k + 1] = np.pi / (k - toggled_frame)
                toggled_frame = k



   #Sometimes StepFrequency is empty, so to prevent nan, we replace nan with zero using nan_to_num
    Step_Frequency[Step_Frequency == 0.0] = np.nan_to_num(Step_Frequency[Step_Frequency != 0.0].mean())
    Gama[i, 0:1] = Step_Frequency.mean(axis=0) #compute average of 4 joints for calculation frequency
    Gama[i, 1:5] = Step_duration

# Make Gamma zmuv
Gmean = Gama.mean(axis=2).mean(axis=0)[np.newaxis, :, np.newaxis]
Gstd = Gama.std(axis=2).mean(axis=0)[np.newaxis, :, np.newaxis]
Gama = (Gama - Gmean) / Gstd
# np.savez_compressed('ZMUV_TauOmega.npz', Wmean=Gmean, Wstd=Gstd)

#Train the network
input_X = Input(shape=(240,3))
Conv_layer = Conv1D(64, 65 , activation='relu', padding='same',use_bias=True)(input_X)
Dropout_layer1 = Dropout(rate=0.2)(Conv_layer)
Conv_layer = Conv1D(5,(45,), activation='linear', padding='same',use_bias=True)(Conv_layer) # the activation function for next layer is linear because we are regressing


Regressor = Model(input_X, Conv_layer)
Regressor.summary()
Regressor.compile(optimizer='adam', loss='mse')

Trajectory = np.swapaxes(Trajectory, 1, 2)
Gama = np.swapaxes(Gama, 1, 2)


T_train = Trajectory[:1500]
T_valid = Trajectory[1500:]
W_train = Gama[:1500]
W_valid = Gama[1500:]

Regressor.fit(T_train, W_train,
                epochs=100,
                batch_size=1,
              validation_data=(T_valid,W_valid),
                callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=True)])

# save model for future use
model_json = Regressor.to_json()
with open(".\\model\\regressor.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
Regressor.save_weights(".\\model\\regressor.h5")
print("Saved model to disk")