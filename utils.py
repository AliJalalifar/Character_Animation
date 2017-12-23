################################################
#                                              #
#           Some Helper Functions              #
#                                              #
################################################

import numpy as np
from keras.models import model_from_json,,Input,Model

# a helper function for loading trained networks
def load_Model(path):
    # load json and create model
    json_file = open(path + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(path + ".h5")
    return loaded_model

# a simple function for extracting foot contact form tau and omega. see 6.3 "Modeling Contact State by Square Waves"
def F_extract(W,c=1,ah=-0.1,at=0,bh=0,bt=0):
    return (np.sign(np.sin(np.cumsum(c * W[:,:, 0:1], axis=1)[0] + ah) - (np.cos(W[:,:, 1:2])[0] + bh))-0.5)/7.73, (np.sign(np.sin(np.cumsum(c * W[:,:, 0:1], axis=1)[0] + at) - (np.cos(W[:,:, 2:3])[0] + bt))-0.5)/7.73,(np.sign(np.sin(np.cumsum(c * W[:,:, 0:1], axis=1)[0] + ah+np.pi) - (np.cos(W[:,:, 3:4])[0] + bh))-0.5)/7.73,(np.sign(np.sin(np.cumsum(c * W[:,:, 0:1], axis=1)[0] + at+np.pi) - (np.cos(W[:,:, 4:5])[0] + bt))-0.5)/7.73


# Encoding X input | feature space -> hidden space
def Motion_Edit(hs_vector):
    loaded_model = load_Model('.\\model\\autoencoder_ND')
    Xinput = Input(shape=(240,70))
    Conv_layer  = loaded_model.layers[1](Xinput)
    Pooling  = loaded_model.layers[2](Conv_layer)
    encoder = Model(Xinput,Pooling)
    return encoder.predict(hs_vector)

