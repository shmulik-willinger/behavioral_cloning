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
## Collecting the Training Set

To capture good driving behavior, I first recorded a couple of laps on track one using center lane driving. Each recording epoch consist of 3 images represent 3 cameras on the vehicle. Here is an example image of center lane driving, from the look of the three cameras:

Left camera              |  Center camera | Right camera
:---------------------:|:---------------------:|:---------------------:
![]( https://github.com/shmulik-willinger/behavioral_cloning/blob/master/readme_img/left_2017_08_13_15_56_31_426.jpg?raw=true)  |  ![]( https://github.com/shmulik-willinger/behavioral_cloning/blob/master/readme_img/center_2017_08_13_15_56_31_426.jpg?raw=true) |  ![]( https://github.com/shmulik-willinger/behavioral_cloning/blob/master/readme_img/right_2017_08_13_15_56_31_426.jpg?raw=true)

To aim for 'center of the road' driving, I recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct wrong behaviors.

Some of the interesting challenges on the tracks including driving in the sharp curves, different texture and different borders of the road, and the second Track is also different from the first one. In order to train the car to drive on all of them I repeated the collection process on all the tracks and laps. Driving in the simulator is pretty difficult in some curves.

After the collection process, I had X number of data points images to start with.

Each image is actually 160 pixels high and 320 pixels wide, with 3 channels -RGB.

I stoped collecting data after observing the overfitting when training the model.

___________________

## Pre-process the Data Set

I needed lots of data in order to well train the model. After collecting all the data from the two tracks, driving back and forth, and using all the 3 cameras images, I also performed the following steps:

The top portion of the image capture sky, trees ans other elements which are unnessery for the training process and might distract the model, the same for the buttom portion of the image. So I croped The images to remove this parts. Each image reduced volume by 50%, and I gained a more accurate model, along with savings space, memory and runtime.
Here is an example of the cropping process on an image:

original image      |  Cropping process | Cropped image
:---------------------:|:---------------------:|:---------------------:
![]( https://github.com/shmulik-willinger/behavioral_cloning/blob/master/readme_img/original.jpg?raw=true)  |  ![]( https://github.com/shmulik-willinger/behavioral_cloning/blob/master/readme_img/cropped.jpg?raw=true) |  ![]( https://github.com/shmulik-willinger/behavioral_cloning/blob/master/readme_img/processed.jpg?raw=true)

Since each recording epoch has 3 images represent 3 cameras on the vehicle, I used all this images to increase the dataset and generalize the model.

To help the model generalize and the accuracy on the opposite curves, I flipped the images and angles to simulate driving counter-clockwise, thinking that this would save me to collect more reliable information and will double the dataset. There are couple of ways to flip an image, I used cv2.flip() after reading that it usually faster that using ather methods like np.fliplr().  here is an image that has been flipped:

original image      |  flipped image
:---------------------:|:---------------------:
![]( https://github.com/shmulik-willinger/behavioral_cloning/blob/master/readme_img/before_flip.jpg?raw=true)  |  ![]( https://github.com/shmulik-willinger/behavioral_cloning/blob/master/readme_img/after_flip.jpg?raw=true)

Using Keras Generator

Using Data Distribution Flattening

Using Normalization and mean zero

I finally randomly shuffled the data set and put 20% of the data into a validation set.

___________________

## Training process

**Training Strategy**

I used the images as the feature set, and the steering measurments as the lables set for training the model.

The overall strategy for deriving a model architecture was to start with a known self-driving car model.

My first step was to use a convolution neural network model similar to the nVidia architecture. I thought this model might be appropriate because it was built and designed as a self-driving car model.

Ideally, the model will make good predictions on both the training and validation sets. The implication is that when the network sees an image, it can successfully predict what angle was being driven at that moment.

**Approach taken for finding the solution**

My model consists of a convolution neural network with 3 Convolution layers with 5x5 filter sizes and depths between 24 and 48, along with 3 MaxPool layers followed by Dropout layers.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.

At the end of the model I have 4 Dense layers and a single output node that will predict the steering angle, which makes this a regression network.

Because this model is a regression network (and not classification) and I what it to minimize the error between the steering measurment that the network predict and the truth steering, so I used 'mean suared error' for the loss function (instead the cross-antropy) since it's a good function for it.

I randomly shuffled the data and split off 20% for a validation set.

After training the model I run the simulator to see how well the car was driving around track one. In the first Attempts the car was swerving from side to side and couldn't drive correctlly, and there were a few spots (esspecialy on sharp curves) where the vehicle fell off the track or even to the lack. To improve the driving behavior in these cases, I took more capture images as a datasets only in the curves. Then I continued tuning parameters and layers values in the model, and run the simulator on the train results to compare the changes.

I noticed that the training loss and validation loss are both high, so I perform the following:
* In order to normalize the data I added a layer in the model to divide each pixel by 255 (max value) which will get the value range between 0 to 1.
* subtracted 0.5 from each element, to mean centering the data.

Both normalizations were done by adding a lambda layer to the model, which is convenient way to parallelize the process

** Parameters tuning **

The model used an adam optimizer, so the learning rate was not tuned manually.

Keras is using 10 epochs as default. After observing that the validation loss decreases for the first 3 epoches and then increasing again, I changed the epochs parameter to 3 to prevent overfitting.


** Reduce overfitting in the model **

The validation set was took as 20% of the dataset and helped determine if the model was over or under fitting.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model and added Dropout layers and more dataset. The model was also trained and validated on different data sets to ensure that the model was not overfitting.


## Model Architecture


####2. Final Model Architecture

My model consisted of the following layers:

<!---
| Layer | Component        	|     Input	      	| Output |
|:---------------------:|:---------------------------------------------:|
| Convolution layer 1 | 2D Convolution layer with 'VALID' padding, filter of (5x5x1) and Stride=1 | (32,32,1) 	| (28,28,6)|
|   	| ReLU Activation 	| (28,28,6) | (28,28,6)|
| Convolution layer 2|	2D Convolution layer with 'SAME' padding, filter of (3x3x6) and Stride=1	|(28,28,6) | (28,28,6)|
|    	| Max pooling	with 'VALID' padding, Stride=2 and ksize=2	| (28,28,6) | (14,14,6)|
| Convolution layer 3   | 2D Convolution layer with 'VALID' padding, filter of (5x5x12) and Stride=1	| (14,14,6)| (10,10,16)|
| 	|  ReLU Activation  		|(10,10,16)|(10,10,16)|
| 	| Max pooling	with 'VALID' padding, Stride=2 and ksize=2	|(10,10,16)|(5,5,16)|
| Fully connected	layer 1	| Reshape and Dropout|(5,5,16)| 400|
| | Linear combination WX+b |400| 120|
| | ReLU and Dropout |120| 120|
| Fully connected	layer 2	| Linear combination WX+b|120| 84|
| | ReLU and Dropout |84| 84|
| Fully connected	Output layer	| Linear combination WX+b|84| 43 |
-->
![]( https://github.com/shmulik-willinger/traffic_sign_classifier/blob/master/readme_img/model.jpg?raw=true)


Here is a visualization of the architecture:


## Results and conclusions

I was satisfied that the model is making good predictions on the training and validation sets, and at the end of the process, the vehicle is able to drive autonomously around Track-1 and Track-2  without leaving the road.

I used the file video.py to record the video of the car while driving autonomously in the simulator, and the video files can be found here as well.

Behavior Cloning is fun :)
