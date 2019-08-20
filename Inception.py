# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.keras.layers.core import Dense, Flatten
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.utils import plot_model
from tensorflow.keras import Model
from tensorflow.python.keras.optimizers import SGD, RMSprop
from constants import github_drive_location

input_img = Input(shape=(150, 150, 3))

### 1st layer
layer_1 = Conv2D(2, (1,1), padding='same', activation='relu')(input_img)
layer_1 = Conv2D(3, (3,3), padding='same', activation='relu')(layer_1)

layer_2 = Conv2D(2, (1,1), padding='same', activation='relu')(input_img)
layer_2 = Conv2D(3, (5,5), padding='same', activation='relu')(layer_2)

layer_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(input_img)
layer_3 = Conv2D(3, (1,1), padding='same', activation='relu')(layer_3)

mid_1 = tf.keras.layers.concatenate([layer_1, layer_2, layer_3], axis = 3)

flat_1 = Flatten()(mid_1)

dense_1 = Dense(600, activation='relu')(flat_1)
dense_2 = Dense(300, activation='relu')(dense_1)
dense_3 = Dense(150, activation='relu')(dense_2)
output = Dense(30, activation='softmax')(dense_3)

my_model = Model([input_img], output)

plot_model(my_model,
           to_file = github_drive_location + '\\InceptionModel.png',
           show_shapes=True,
           show_layer_names=True)

# Print the model summary
my_model.summary()

def list_layers():
    first_five = my_model.layers[: 5]
    #last_five = my_model.layers[-5 :]
    for layer in first_five:
        class_of_layer = str(layer)[0 : str(layer).index(" ")]
        print(class_of_layer[class_of_layer.rfind('.') + 1 : ] + " - " + layer.name)

a = list_layers()

my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

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
model.summary()
'''