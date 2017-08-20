
#Loading the data
import os 
import csv
import numpy as np
import cv2
import random
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Model

lines=[]
data_folder = 'C:\data\\'
file = '\driving_log.csv'
for child in os.listdir(data_folder):
    data_subfolder = os.path.join(data_folder, child)
    sub_file = data_subfolder + file
    with open(sub_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    print ("file: ", sub_file, ", data samples: ", len(lines))
            
print("total data samples: ", len(lines))


def flip_images(images, measurements):
    flipped_images, flipped_measurements = [], []
    for img, measurements in zip(images, measurements):
        flipped_images.append(cv2.flip(img, 1))
        flipped_measurements.append(measurements * -1.0)
    return flipped_images, flipped_measurements
        
       
def brightness_randomization(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[::2] = hsv[::2] * (.5 + np.random.random())
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def crop_images(images):
    cropped_images= []
    for img in images:
        cropped_images.append(img[60:140, 0:320])
    return cropped_images


# Spliting the dataset to Train and Validation
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)


angle_deviation=0.05

def load_images(line):
    images = []
    measurements = []
    # Load the center image and steering
    steering = float(line[3])
    images.append(cv2.imread(line[0]))
    measurements.append(steering)

    # For 25% of the cases - load also the right and left images
    if (random.randint(1, 4) == 4):
        images.append(cv2.imread(line[1]))
        measurements.append(steering + angle_deviation)
        images.append(cv2.imread(line[2]))
        measurements.append(steering - angle_deviation)

    # Adding flipped images for 30% of the cases
    if (random.randint(1, 3) == 3):
        flipped_images, flipped_measurements = flip_images(images, measurements)
        images.extend(flipped_images)
        measurements.extend(flipped_measurements)
    return images, measurements



def generator(samples, batch_size=32):
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, len(samples), batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images, measurements = [], []
            for batch_sample in batch_samples:
                loaded_images, loaded_imeasurements = [],[]
                #Loading the original images from the 3 cameras, cropping, flipping and append to the dataset
                loaded_images, loaded_imeasurements = load_images(line)
                cropped_images = crop_images(loaded_images)
                #gray_images = grayscale(cropped_images)
                images.extend(cropped_images)
                measurements.extend(loaded_imeasurements)              
                  
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

print("features Shape: ", train_generator)
print("lables Shape: ", validation_generator)


dropout = 0.5

model = Sequential()
#normalization and mean zero - centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 -1.0, input_shape=(80,320,3))) 
model.add(Conv2D(24, kernel_size=(5, 5), padding='valid', activation='relu'))
model.add(Conv2D(36, kernel_size=(5, 5), padding='valid', activation='relu'))
model.add(Dropout(dropout))
model.add(Conv2D(48, kernel_size=(5, 5), padding='valid', activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(dropout))
model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


from time import time
start_time = time()

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, steps_per_epoch= len(train_samples)/64, 
    validation_data=validation_generator, validation_steps=len(validation_samples), epochs=2, verbose = 1)

# save the model to use it in the simulator
model.save('model.h5')

total_time = time() - start_time
minutes, seconds = divmod(total_time, 60)
print ("Total time for training: ", minutes, "min, {:.0f}".format(seconds),  "s ")

