# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()

from constants import external_drive_location
import callback as callback_file
from split_data_training_test import split_data_training_test as split_data
from categorize_data import categorize_data as categorize
from display_images import display_images as display
from data_generators import train_generator, validation_generator
from Inception import my_model

'''
split = split_data(data_directory)
split.move_training_to(training_directory)
print("Data split into training directory")
split.move_test_to(testing_directory)
print("Data split into test directory")

categories = categorize(data)
categories.in_directory(training_directory)
print("Data categorized in the training directory")
categories.in_directory(testing_directory)
print("Data categorized in the testing directory")

display("buffalo", training_directory).numberOfTimes(4)
'''

# Fit the model using the training and validation generators, specify the number of 
# epochs to train for and how descriptive the output should be
try:
    history = my_model.fit_generator(
    train_generator,
    validation_data = validation_generator,
    steps_per_epoch = 50,
    epochs = 20,
    validation_steps = 50,
    verbose = 1,
    callbacks=[callback_file.myCallback()])
except:
    print("Training interrupted, saving model")
    my_model.save_weights(external_drive_location + "\\V3_Parthiv_Attempt_1_Interrupt.h5")
    
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()