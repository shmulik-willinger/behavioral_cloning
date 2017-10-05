# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
The goal of this project is to effectively train a car to drive autonomously in a simulator. The simulator provided by Udacity and has two different tracks the car can drive in.

In order to train the model, the project is using deep neural networks and convolutional neural networks to clone driving behavior. The model Using Keras framework over Tensorflow and output a steering angle to the autonomous vehicle.

A detailed description of the project including the model, data and visualizing is also provided  [here](https://github.com/shmulik-willinger/behavioral_cloning/blob/master/writeup.md)


![]( https://github.com/shmulik-willinger/behavioral_cloning/blob/master/readme_img/sim-image.jpg?raw=true)

The Goals
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report


## Details About the Files

The project includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* Behavioral_Cloning.ipynb - the notebook with the data preprocessing and the model training
* model.py - the script used to create and train the model
* drive.py - for driving the car in autonomous mode
* model.h5 - containing the trained convolution neural network
* model.json - the architecture of the model as json
* writeup.md - summarizing the project and results
* video.mp4 - a video recording of the vehicle driving autonomously around the track

The simulator can be downloaded from [here](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip)

 The car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

The Behavioral_Cloning.ipynb file contains the code for training and saving the convolution neural network. The notebook shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. The sections are divided according to:
* Gathering the data
* Preprocessing the data by batchs
* Defining the Model
* Training the Model

## Output video

The output video of the car completing the tracks can be found here:

Track 1  |  Track 2 (partially)
:-------------------------:|:-------------------------:
[![video track_1](https://github.com/shmulik-willinger/behavioral_cloning/blob/master/readme_img/behavioral_cloning_simulator_track_1.gif)](http://www.youtube.com/watch?v=fIvBNRCIY4U)  |  [![video track_2](https://github.com/shmulik-willinger/behavioral_cloning/blob/master/readme_img/behavioral_cloning_simulator_track_2.gif)](http://www.youtube.com/watch?v=A1280XlpITA)


## Dependencies
This project requires **Python 3.5** and the following Python libraries installed:

- [Jupyter](http://jupyter.org/)
- [NumPy](http://www.numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [OpenCV](https://pypi.python.org/pypi/opencv-python#)
- [Sklearn](scikit-learn.org/)
- [Pandas](pandas.pydata.org/)
- [TensorFlow](http://tensorflow.org) version 1.2.1
- [Keras](https://keras.io/) version 2.0.6
