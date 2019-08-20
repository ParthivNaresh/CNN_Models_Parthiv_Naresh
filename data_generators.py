# -*- coding: utf-8 -*-

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from constants import training_directory, testing_directory
'''

'''

# Rescaling and augementations for the training data
training_datagen = ImageDataGenerator(
        rescale = 1./255.,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
# Rescaling for the test data
validation_datagen = ImageDataGenerator(rescale = 1./255.)
  
# Training and test generators label data based on the folder name
train_generator = training_datagen.flow_from_directory(
        # specify the output size and type of classification (binary, category, etc)
        training_directory,
        batch_size = 26,
        target_size=(150,150),
        class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
        testing_directory,
        batch_size = 26,
        target_size=(150,150),
        class_mode='categorical')