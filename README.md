# What is This
this is a simple re-implementation of the paper "A Deep Learning Framework for Character Motion Synthesis and Editing"(1). Only Sections 5 and 6 are re-implemented.


# Demo
To see a demo, download "Demo.mp4" or simply run "Demo.py". To run correclty, Keras with tensorflow backend is required.

# Structure
Autoencoder.py learns the motion manifold using CNN. This is the re-implementation of section 5.
Motion_Synthesis.py maps trajectory and foot contact information to motion in hidden space. This is the re-implementation of section 6.2.
RegressTauOmega.py learns a regresseion between trajectory and step frequency/duration for disambiguation. This is the re-implementation of section 6.3.

The input to the system is a 3 dimensional vector which describes the trajectory of the movement. Then the data of
setp frequency/duration is extracted from trajectory and converted to foot contact information. Later, we feed this data to the
Motion Synthesis network which creates motion in hidden space. Finally, by using decoder part of auto encoder, a low-level
description of the movement is achieved.

*Notice that to re-train the network, you shoud place the processed CMU dataset in "\data" folder. Due to it's huge size, it's omitted.

# Database
The data used in this project was obtained from mocap.cs.cmu.edu.
The database was created with funding from NSF EIA-0196217.
[CMU. Carnegie-Mellon Mocap Database](http://mocap.cs.cmu.edu/)

# Refrences

[1] Holden D, Saito J, Komura T. A deep learning framework for character motion synthesis and editing. ACM Transactions on Graphics (TOG). 2016 Jul 11;35(4):138.
[2] Holden D, Saito J, Komura T, Joyce T. Learning motion manifolds with convolutional autoencoders. InSIGGRAPH Asia 2015 Technical Briefs 2015 Nov 2 (p. 18). ACM.
[3] CMU. Carnegie-Mellon Mocap Database. http://mocap.cs.cmu.edu/.
