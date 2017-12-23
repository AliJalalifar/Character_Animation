################################################
#                                              #
#      Make Data Zero-mean/Unit-variance       #
#                                              #
################################################

import numpy as np

Dataset_address = '.\\data\\data_cmu2.npz'

#loading data from cmu datset
X = np.load(Dataset_address)['clips']
X = np.swapaxes(X, 1, 2)

#calculation mean for every feature
mean = X.mean(axis=2).mean(axis=0)

#calculation std for every feature
std = X.std(axis=2).std(axis=0)

#save data for future use
np.savez_compressed('ZMUV_locomotion.npz', mean=mean, std=std)
