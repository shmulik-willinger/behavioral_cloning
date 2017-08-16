# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
The goal of this project is to effectively train a car to drive autonomously in a simulator. The simulator provided by Udacity and has two different tracks the car can drive in.

In order to train the model, the project is using deep neural networks and convolutional neural networks to clone driving behavior. The model Using Keras framework and output a steering angle to the autonomous vehicle.

A detailed description of the project including the model, data and visualizing is also provided  [here](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md)


![]( https://github.com/shmulik-willinger/behavioral_cloning/blob/master/readme_img/sim-image.png?raw=true)

The Goals
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report


## Details About Files & Code Quality

The project includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* behavioral_cloning.ipynb - the notebook with the data preprocessing and the model training
* drive.py - for driving the car in autonomous mode
* model.h5 - containing a trained convolution neural network
* writeup.md - summarizing the project and results
* video.mp4 - a video recording of the vehicle driving autonomously around the track

Submission includes functional code
Using the Udacity provided simulator and my drive.py file:
```sh
[Download the simulator](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/46a70500-493e-4057-a78e-b3075933709d/concepts/1c9f7e68-3d2c-4313-9c8d-5a9ed42583dc)
```
 The car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

Submission code is usable and readable

The behavioral_cloning.ipynb file contains the code for training and saving the convolution neural network. The notebook shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. The sections are divided according to:
* Gathering the data
* preprocessing the data by batchs
* Defining the Model
* Training the Model

## Dependencies
This project requires **Python 3.5** and the following Python libraries installed:

- [Jupyter](http://jupyter.org/)
- [NumPy](http://www.numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [OpenCV](https://pypi.python.org/pypi/opencv-python#)
- [Sklearn](scikit-learn.org/)
- [Pandas](pandas.pydata.org/)
- [TensorFlow](http://tensorflow.org)
