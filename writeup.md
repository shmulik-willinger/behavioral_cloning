# **Behavioral Cloning**

**Train a car to drive a simulator autonomously**


## The goals

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report




---
## Creation of the Training Set

To capture good driving behavior, I first recorded a couple of laps on track one using center lane driving. Each recording epoch consist of 3 images represent 3 cameras on the vehicle. Here is an example image of center lane driving, from the look of the three cameras:

Left camera              |  Center camera | Right camera
:---------------------:|:---------------------:|:---------------------:
![]( https://github.com/shmulik-willinger/behavioral_cloning/blob/master/readme_img/left_2017_08_13_15_56_31_426.jpg?raw=true)  |  ![]( https://github.com/shmulik-willinger/behavioral_cloning/blob/master/readme_img/center_2017_08_13_15_56_31_426.jpg?raw=true) |  ![]( https://github.com/shmulik-willinger/behavioral_cloning/blob/master/readme_img/right_2017_08_13_15_56_31_426.jpg?raw=true)

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct wrong behaviors. These images show what a recovery looks like:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Different tracks have different surface, and even in the same track there are a couple surface structure. In order to train the car to drive on all of them I repeated the collection process on all the tracks and laps.

To augment the data set, I also flipped the images and angles thinking that this would save me to collect more reliable information, and also will help the model accuracy on the opposite curves, here is an image that has been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had X number of data points. I then preprocessed this data.

___________________

## Pre-process the Data Set

I needed lots of data in order to well train the model. After collecting all the data from the two tracks, driving back and forth, and using all the 3 cameras images, I also performed the following:

The model will not have to process the background of th images and the front of the car which appears in every image, so I used cropping in order to reduce the images volume by 60% ! By using this simple code line:
```sh
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
```
original image      |  Cropping process | Processed image
:---------------------:|:---------------------:|:---------------------:
![]( https://github.com/shmulik-willinger/behavioral_cloning/blob/master/readme_img/original.jpg?raw=true)  |  ![]( https://github.com/shmulik-willinger/behavioral_cloning/blob/master/readme_img/cropped.jpg?raw=true) |  ![]( https://github.com/shmulik-willinger/behavioral_cloning/blob/master/readme_img/processed.jpg?raw=true)

Using Keras Generator

Using Data Distribution Flattening

Using Normalization and mean zero

I finally randomly shuffled the data set and put 20% of the data into a validation set.

___________________

## Model Architecture

**Approach taken for finding the solution**

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the graph attached below in the model training I used an adam optimizer so that manually training the learning rate wasn't necessary.

The overall strategy for deriving a model architecture was to start with a known self-driving car model.

My first step was to use a convolution neural network model similar to the nVidia architecture. I thought this model might be appropriate because it was built and designed as a self-driving car model.



My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24)

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18).

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ...

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach



In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that ...

Then I ...

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
