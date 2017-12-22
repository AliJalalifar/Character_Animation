# What is This
This is a simple re-implementation of the paper "A Deep Learning Framework for Character Motion Synthesis and Editing"(1). Only Sections 5 and 6 are re-implemented.


# Demo
To see a demo, download "Demo.mp4" or simply run "Demo.py". To run correclty, Keras with tensorflow backend is required.

# Structure
**Autoencoder.py** learns the motion manifold using CNN. This is the re-implementation of section 5.<br/>
**Motion_Synthesis.py** maps trajectory and foot contact information to motion in the hidden space. This is the re-implementation of section 6.2.<br/>
**RegressTauOmega.py** learns a regresseion between trajectory and step frequency/duration for disambiguation. This is the re implementation of section 6.3.<br/>
**Demo.py** randomly select a curve from the file "data\curvez.npz" and create the character animation with respect to the curve. These curves are not used during training process.<br/>
The input to the system is a 3 dimensional vector which describes the trajectory of the movement. Then the data of step frequency/duration is extracted from trajectory and converted to foot contact information. Later, we feed this data to the Motion Synthesis network which creates motion in hidden space. Finally, by using decoder part of the autoencoder, a low-level description of the movement is achieved.<br/>
*Notice that to re-train the network, you shoud place the processed CMU dataset in "\data" folder. Due to it's huge size, it's not included.

# Database
The data used in this project was obtained from mocap.cs.cmu.edu. <br />
The database was created with funding from NSF EIA-0196217.<br />
[CMU. Carnegie-Mellon Mocap Database](http://mocap.cs.cmu.edu/)

# References

[1] Holden D, Saito J, Komura T. A deep learning framework for character motion synthesis and editing. ACM Transactions on Graphics (TOG). 2016 Jul 11;35(4):138.<br />
[2] Holden D, Saito J, Komura T, Joyce T. Learning motion manifolds with convolutional autoencoders. InSIGGRAPH Asia 2015 Technical Briefs 2015 Nov 2 (p. 18). ACM.<br />
[3] CMU. Carnegie-Mellon Mocap Database. http://mocap.cs.cmu.edu/.

