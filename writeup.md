# **Behavioral Cloning**

**Train a car to drive autonomously in a simulator**


## The goals

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report




---
## Collecting the Training Set

To capture good driving behavior, I first recorded a couple of laps on track one using center lane driving. Each recording epoch consist of 3 images represent 3 cameras on the vehicle. Having 3 images for each epoch help collecting more data, and teach the network how to steer back to the center and correct wrong behaviors.

Here is an example of how center lane driving looks from all the three cameras:

Left camera              |  Center camera | Right camera
:---------------------:|:---------------------:|:---------------------:
![]( https://github.com/shmulik-willinger/behavioral_cloning/blob/master/readme_img/left_2017_08_13_15_56_31_426.jpg?raw=true)  |  ![]( https://github.com/shmulik-willinger/behavioral_cloning/blob/master/readme_img/center_2017_08_13_15_56_31_426.jpg?raw=true) |  ![]( https://github.com/shmulik-willinger/behavioral_cloning/blob/master/readme_img/right_2017_08_13_15_56_31_426.jpg?raw=true)

In order to use the left and right images I needed to add a correction factor for the center-steering measurment, in order to  create adjusted steering measurements for the side camera images

Some of the interesting challenges on the tracks includes driving in the sharp curves, different texture and different borders of the road, and the second Track is also different from the first one. In order to train the car to drive on all of them I repeated the collection process on all the tracks and laps. Driving in the simulator is pretty difficult in some curves.

I collected data of center lane driving from 5 laps, and some extra data for 'recovery driving from the sides', along with special data recording driving smoothly around 10 sharp curves.

After the collection process, I had 74760 data points images to start with.

Each image dimensions is 160 X 320 pixels with 3 channels -RGB.

I stoped collecting data after observing the overfitting when training the model.

___________________

## Pre-process the Data Set

Data augmentation has a couple of benefits, including adding more comprehensive data for training.
I needed lots of data in order to well train the model. After collecting all the data from the two tracks, driving back and forth, and using all the 3 cameras images, I also performed the following steps:

The top portion of the image capture sky, trees and other elements which are unnecessary for the training process and might distract the model, the same for the lower portion of the image. So I decided to remove pixels containing redundant information. I cropped The images to remove this pixels. Each image reduced capacity by 50%, and I gained a more accurate model, along with savings space, memory and model runtime.

Here is an example of the cropping process on an image:

original image      |  Cropping process | Cropped image
:---------------------:|:---------------------:|:---------------------:
![]( https://github.com/shmulik-willinger/behavioral_cloning/blob/master/readme_img/original.jpg?raw=true)  |  ![]( https://github.com/shmulik-willinger/behavioral_cloning/blob/master/readme_img/cropped.jpg?raw=true) |  ![]( https://github.com/shmulik-willinger/behavioral_cloning/blob/master/readme_img/processed.jpg?raw=true)

Since each recording epoch has 3 images represent 3 cameras on the vehicle, I used all this images to increase the dataset and generalize the model. It's important to add a correction-angle  for the steering measurement for the left and right cameras, since the steering provided from the simulator represent the center camera only.

 ![]( https://github.com/shmulik-willinger/behavioral_cloning/blob/master/readme_img/correction-angle.png?raw=true)

To help the model generalize better and to improve the accuracy on the opposite curves, I flipped the images and angles to simulate driving counter-clockwise, thinking that this would save me from collecting more reliable information and will double the dataset. There are couple of ways to horizontally flip an image, I used cv2.flip() after observing that it is much faster than using other methods like np.fliplr().

Here is an example of image that has been flipped in that process:

original image      |  flipped image
:---------------------:|:---------------------:
![]( https://github.com/shmulik-willinger/behavioral_cloning/blob/master/readme_img/before_flip.jpg?raw=true)  |  ![]( https://github.com/shmulik-willinger/behavioral_cloning/blob/master/readme_img/after_flip.jpg?raw=true)

During the training I have tried some other image processing techniques like reduced the bandwidth of the images by converting each image from BGR to YUV, adding brightness randomization by convering images to HSV and more. After I realized that these changes were not conducive to better prediction outcomes - I removed them from the final preprocessing step.

Here is an example of some image transformation I tried:

reduce bandwith      |  brightness randomization | grayscale
:---------------------:|:---------------------:|:---------------------:
![]( https://github.com/shmulik-willinger/behavioral_cloning/blob/master/readme_img/reduce_bandwith.jpg?raw=true)  |  ![]( https://github.com/shmulik-willinger/behavioral_cloning/blob/master/readme_img/brightness_randomization.jpg?raw=true)|  ![]( https://github.com/shmulik-willinger/behavioral_cloning/blob/master/readme_img/grayscale.jpg?raw=true)

Regarding the distribution of the steering lables, It is clear to see that steering angle s=0 have the highest frequency, and that there are more positive angles than negative, meaning the dataset is not balance (By horizontally reversing the images we gain distribution balancing)

![]( https://github.com/shmulik-willinger/behavioral_cloning/blob/master/readme_img/steering_distribution.jpg?raw=true)

___________________

## Training process

**Training Strategy**

I used the images as the feature set, and the steering measurments as the labels set for training the model.

The overall strategy was to start with a known self-driving car model, so I used a convolution neural network model similar to the [nVidia architecture](https://github.com/shmulik-willinger/behavioral_cloning/blob/master/readme_img/nvidia.jpg). I thought this model might be appropriate because it was built and designed as a self-driving car model. The original model architecture looks as Follow:

 ![]( https://github.com/shmulik-willinger/behavioral_cloning/blob/master/readme_img/nvidia.png?raw=true)

Ideally, the model will make good predictions on both the training and validation sets. The implication is that when the network sees an image, it can successfully predict what angle was being driven at that moment.

**Approach taken for finding the solution**

Because this model is a regression network (and not classification) I what it to minimize the error between the steering measurment that the network predict and the truth steering, so I used 'mean squared error' for the loss function (and not the cross-antropy) since it's a good function for it.

I randomly shuffled the data and split off 20% for a validation set.

After training the model I ran the simulator to see how well the car was driving around track one. In the first attempts the car was swerving from side to side and couldn't drive correctlly, and there were a few spots (esspecialy on sharp curves) where the vehicle fell off the track. To improve the driving behavior in these cases, I took more capture images as a datasets only in the curves. Then I continued tuning parameters and layers values in the model, and ran the simulator on the train results to compare the changes.

I noticed that the training loss and validation loss are both high, so I perform the following:
* In order to normalize the data I added a layer in the model to divide each pixel by 255 (max value) which will get the value range between 0 to 1.
* subtracted 0.5 from each element, to mean centering the data.

Both normalizations were done by adding a lambda layer to the model, which is convenient way to parallelize the process

Here we can see the training and validation 'mean squared error loss' during the training:

![]( https://github.com/shmulik-willinger/behavioral_cloning/blob/master/readme_img/mse_loss.png?raw=true)


**Parameters tuning**

To add the left and right images for each epoch I needed to configure the 'correction factor' parameter to set their steering value. I started with the default 0.2 and made some experiments to changes it till I noticed an improvment in the driving simulator.

The model used an adam optimizer, so the learning rate was not tuned manually.

Keras is using 10 epochs as default. After observing that the validation loss decreases for the first 3 epoches and then increasing again, I changed the epochs parameter to 3 to prevent overfitting.



**Reduce overfitting in the model**

In order to gauge how well the model was working, I split my images and steering angle data into a training and validation set. The validation set took 20% of the dataset and helped determine if the model was over or under fitting.

I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To deal with the overfitting, I modified the model and added Dropout layers and more dataset. (see architecture section below)

**Using Generator**

The images captured in the car simulator are very large comparing to dataset images for other common networks. Each image contains 76,800 pixels (80X320X3) after cropping, and when the dataset contains 70K images we need a huge memory for network training. I used Generator, enables to train the model by producing batches with data processing in real time, only when the model need it.

The disadvantage of using the generator is that errors are being hidden inside the thread, and if the inside code (generator function) is failing for any exception such as loading the images, cropping etc., the file output will be 'StopIteration' with no more information.

**Remarks**

When running the simulator to drive autonomously, it's running the model.prediction() function that gets the current image from the center camera. In order to get the right prediction it's necessary to process the image the same steps as we ran on the training set, including cropping and color changing. Missing this step will lead to strange predictions in the simulator.

Machine learning involves trying out ideas and testing them to see if they work. If the model is over or underfitting, then try to figure out why and adjust accordingly. I tried a couple of ideas like reduced the bandwidth of the image by converting each image from BGR to YUV, adding brightness randomization by convering images to HSV, adding more layers (Fully connected, Polling, Conv2D and more) and checking each time how if affects the prediction in the simulator.

I had a couple of issues with the version of Keras and Tensorflow. Since the training was performed on an Amazon g2.2xlarge GPU server, and the simulator driving was running on my local PC. In order for the model.h5 file to run correctly the TensorFlow and Keras must be exectly in the same version (latest).

OpenCV read images as BGR, and the file drive.py (to run the car in the simulator) uses RGB. In order for the images comming from the simulator in the autonomous mode to be process correctly - I used cv2.COLOR_BGR2RGB on them.

## Model Architecture


**Final Model Architecture**

My model consisted of the following layers:


| Layer | Component    	|     Output	 	| # Param |
|:----------------:|:------------:|:------------:|:------------:|
| Lambda | Normalization and mean zero | (None, 80, 320, 3) | 0 |
| Convolution | kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu' | (None, 38, 158, 24) | 1824 |
| Dropout	| rate=0.5 | (None, 38, 158, 24) | 0 |
| Convolution |	kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'	|(None, 17, 77, 36) | 21636 |
| Dropout	| rate=0.5 | (None, 17, 77, 36) | 0 |
| Convolution |kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'	| (None, 7, 37, 48) | 43248 |
| Dropout	| rate=0.5 | (None, 7, 37, 48) | 0 |
| Convolution	| kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu' | (None, 5, 35, 64) | 27712|
| Dropout	| rate=0.5 | (None, 5, 35, 64) | 0 |
| Convolution	| kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu' | (None, 3, 33, 64) | 36928 |
| Dropout	| rate=0.5 | (None, 3, 33, 64) | 0 |
| Flatten | Flattens the input to one dimension |(None, 6336)| 0 |
| Dense | Fully connected |(None, 100)| 633700|
| Dense | Fully connected |(None, 50)| 5050|
| Dense | Fully connected |(None, 10)| 510|
| Dense | Output layer  |(None, 1)| 1|

The network consists of a convolution neural network starting with normalization layer, followed by 3 convolution layers with kernel of 5x5 and 2 convolution layer with kernel of 3X3. filter sizes and depths between 24 and 64, with by Dropout layers between them.

The model uses RELU activation on the layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.

At the end of the model I have 4 Fully connected layers and a single output node that will predict the steering angle (regression network)


## Results and conclusions

I was satisfied that the model is making good predictions on the training and validation sets, and at the end of the process, the vehicle is able to drive autonomously around Track-1 and partially on Track-2 without leaving the road.

I used the file video.py to record the video of the car while driving autonomously in the simulator (Consists of the images processed from the simulator on runtime).

The output video of the car completing the tracks can also be found here:

Track 1  |  Track 2 (partially)
:-------------------------:|:-------------------------:
[![video track_1](https://github.com/shmulik-willinger/behavioral_cloning/blob/master/readme_img/behavioral_cloning_simulator_track_1.gif)](http://www.youtube.com/watch?v=fIvBNRCIY4U)  |  [![video track_2](https://github.com/shmulik-willinger/behavioral_cloning/blob/master/readme_img/behavioral_cloning_simulator_track_2.gif)](http://www.youtube.com/watch?v=A1280XlpITA)

Behavior Cloning is fun :+1:
