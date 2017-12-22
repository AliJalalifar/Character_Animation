################################################
#                                              #
#                    Demo                      #
#                                              #
################################################

from utils import load_Model,F_extract
import numpy as np
from AnimationPlot import animation_plot
from keras.layers import Input
from keras.models import Model


preprocess = np.load('ZMUV_locomotion.npz')
preprocess_footcontact = np.load('ZMUV_TauOmega.npz')

Xmean = np.swapaxes(preprocess['Xmean'],1,2)
Xstd = np.swapaxes(preprocess['Xstd'],1,2)

# randomly select a curve from file curvez.npz with frame length of 240.
# these are test curves and the model has never seen them during the training
# phase
k=np.random.random_integers(low=0,high=23160)
Sample_Trajectory = np.load('.\\data\\curves.npz')['C'][:,:,k:k+240]

#preprocessing trajectory
Trajectory = np.swapaxes(Sample_Trajectory, 1, 2)
Trajectory = np.reshape(Trajectory, newshape=(-1, 240, 3))
Trajectory = (Trajectory - Xmean[:, :, -7:-4]) / Xstd[:, :, -7:-4]

#load networks
regressor = load_Model('.\\model\\regressor') #Compute [omega,tau_lh,tau_lt,tau_rh,tau_rt] from Trajectory
motion_synthesis = load_Model('.\\model\\motionsynthesis') #Compute motion in the hidden space from [Trajectory,footcontact]
autoencoder = load_Model('.\\model\\autoencoder_ND') #AutoEncoder for decoding the motion in hidden space

Wmean = np.swapaxes(preprocess_footcontact['Wmean'], 1, 2)
Wstd = np.swapaxes(preprocess_footcontact['Wstd'], 1, 2)


W = regressor.predict(Trajectory) #outputs [omega,tau_lh,tau_lt,tau_rh,tau_rt]
W = (W * Wstd) + Wmean

foot_contact = np.swapaxes(F_extract(W),0,2)[0] #Extract foot_contact information from omega and tau

#concatenation trajectory and foot contact information to feed the decoder of autoencoder
Upsilon_T = (np.concatenate([Trajectory[0], foot_contact], axis=1))
Upsilon_T = np.reshape(Upsilon_T,(1,240,7))

# loading decoder
encoded_input = Input(shape=(120,256))
decoder_layer1 = autoencoder.layers[-3](encoded_input)
decoder_layer2 = autoencoder.layers[-2](decoder_layer1)
decoder_layer3 = autoencoder.layers[-1](decoder_layer2)
decoder = Model(encoded_input, decoder_layer3)

# motion_synthesis outputs motion in hidden space from input Upsilon_T
hs_vector = motion_synthesis.predict(Upsilon_T)

# motion manifold -> feature space
X = decoder.predict(hs_vector)

# inserting the original trajectory to the output of network 'X'
X[:,:,66:69]= Trajectory[0, :, 0:3]


X2 = (X*Xstd)+Xmean
X2 = np.swapaxes(X2,1,2)

# Animate! :D
animation_plot([X2],interval=1)

