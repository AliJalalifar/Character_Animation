import numpy as np
import tensorflow as tf
from keras import Model, Input
import sys
sys.path.append('.\\motion\\')
from motion.view_triple import animation_plot
from utils import load_Model, Gram, Motion_Edit


# Load autoencoder which is trained on both style and content clips
autoencoder = load_Model('.\\model\\autoencoder_combined')

# Load content and style clips
content = np.load('.\\data\\content_cmu.npz')['clips']
style = np.load('.\\data\\style_cmu.npz')['clips']

# Preprocessing
preprocess = np.load('ZMUV_combined.npz')
clip_mean = preprocess['Xmean']
clip_std = preprocess['Xstd']
clip = (content - clip_mean) / clip_std
style = (style-clip_mean)/clip_std

# Select a clip from content dataset and a clip from style dataset
selected_content = clip[250,:,:].reshape(-1,240,73)
selected_style = style[1200,:,:].reshape(-1,240,73)

# Feature space -> Hidden Space
phi_C = Motion_Edit(selected_content,'.\\model\\autoencoder_combined')
phi_S = Motion_Edit(selected_style,'.\\model\\autoencoder_combined')
phi_C = phi_C.reshape(120,256)
phi_S = phi_S.reshape(120,256)

#Compute Gram Matrix of phi_S
Gphi_S = Gram(phi_S)


s=0.1
c=0.9

# Learning Parameters
beta1 = 0.9
beta2 = 0.999
epsilon= 1e-8
learning_rate = 0.01
training_epochs = 1000
display_step = 50

initial_values = np.random.normal(size=(120,256)).astype('float64')
X = tf.Variable(initial_values)
Y = tf.placeholder('float64')
Z = tf.placeholder('float64')
Gx = tf.matmul(tf.transpose(X),X) #Gram matrix of X

# Cost function to be minimized. See 7.2., e.q. 14
cost = s* tf.norm(Y-Gx)+ c* tf.norm(Z-X)

# Gradient descent
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1,beta2=beta2
                                   ,epsilon=epsilon).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        sess.run(optimizer, feed_dict={Y:Gphi_S,Z:phi_C})
# Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={Y:Gphi_S,Z:phi_C})
            print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
    Combined = sess.run(X)


# Loading decoder
encoded_input = Input(shape=(120,256))
decoder_layer1 = autoencoder.layers[-3](encoded_input)
decoder_layer2 = autoencoder.layers[-2](decoder_layer1)
decoder_layer3 = autoencoder.layers[-1](decoder_layer2)
decoder = Model(encoded_input, decoder_layer3)


# Motion manifold -> Feature space
Combined = np.reshape(Combined,(-1,120,256))
X = decoder.predict(Combined)

# Inserting the original trajectory to the output of network 'X'
X[:,:,66:69]= selected_content[0, :, 66:69]

# Output clip, a combination of content clip and style clip
X2 = (X*clip_std)+clip_mean
X2 = np.swapaxes(X2,1,2)

# ground-truth style movement
selected_style = (selected_style*clip_std)+clip_mean
selected_style = np.swapaxes(selected_style,1,2)

# grand-truth content movement
selected_content = (selected_content*clip_std)+clip_mean
selected_content = np.swapaxes(selected_content,1,2)

# Animate! :D
animation_plot([selected_style,X2,selected_content],interval=1)

