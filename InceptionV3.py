# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from constants import inceptionv3_weights
from constants import github_drive_location
'''
pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                include_top = False, 
                                weights = None)

pre_trained_model.load_weights(inceptionv3_weights)

# Make all the layers in the pre-trained model non-trainable
for layer in pre_trained_model.layers:
  layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output
  
x = Flatten()(last_output)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(30, activation='softmax')(x)

my_model = Model(pre_trained_model.input, predictions)
'''
model = tf.keras.models.Sequential([
        # The input shape is the desired size of the image 150x150 with 3 bytes color
        # This is the first convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # The third convolution
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # The fourth convolution
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        #tf.keras.layers.Dropout(0.5),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(30, activation='softmax')])

plot_model(model,
           to_file = github_drive_location + '\\OriginalModel.png',
           show_shapes=True,
           show_layer_names=True)

# Compile the model and specify the loss function, optimizer, and metrics to track
model.compile(optimizer = RMSprop(lr=0.01), 
              loss = 'categorical_crossentropy',
              metrics=['acc'])

# Print the model summary
model.summary()

def list_layers():
    first_five = model.layers[: 5]
    #last_five = my_model.layers[-5 :]
    for layer in first_five:
        class_of_layer = str(layer)[0 : str(layer).index(" ")]
        print(class_of_layer[class_of_layer.rfind('.') + 1 : ] + " - " + layer.name)

a = list_layers()
