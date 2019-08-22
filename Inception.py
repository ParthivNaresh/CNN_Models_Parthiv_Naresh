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

layer_1 = Conv2D(16, (3,3), activation='relu')(input_img)
layer_2 = MaxPooling2D((2,2))(layer_1)

layer_3 = Conv2D(32, (3,3), activation='relu')(layer_2)
layer_4 = MaxPooling2D((2,2))(layer_3)
    
### 1st layer
layer_1_inception = Conv2D(12, (1,1), padding='same', activation='relu')(layer_4)
layer_2_inception = Conv2D(12, (3,3), padding='same', activation='relu')(layer_1_inception)

layer_3_inception = Conv2D(12, (1,1), padding='same', activation='relu')(layer_4)
layer_4_inception = Conv2D(12, (5,5), padding='same', activation='relu')(layer_3_inception)

layer_5_inception = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_4)
layer_6_inception = Conv2D(12, (1,1), padding='same', activation='relu')(layer_5_inception)

mid_1 = tf.keras.layers.concatenate([layer_2_inception, layer_4_inception, layer_6_inception], axis = 3)

layer_5 = Conv2D(36, (3,3), activation='relu')(mid_1)
layer_6 = MaxPooling2D((2,2))(layer_5)

layer_7 = Conv2D(48, (3,3), activation='relu')(layer_6)
layer_8 = MaxPooling2D((2,2))(layer_7)
    
### 1st layer
layer_7_inception = Conv2D(16, (1,1), padding='same', activation='relu')(layer_8)
layer_8_inception = Conv2D(16, (3,3), padding='same', activation='relu')(layer_7_inception)

layer_9_inception = Conv2D(16, (1,1), padding='same', activation='relu')(layer_8)
layer_10_inception = Conv2D(16, (5,5), padding='same', activation='relu')(layer_9_inception)

layer_11_inception = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_8)
layer_12_inception = Conv2D(16, (1,1), padding='same', activation='relu')(layer_11_inception)

mid_2 = tf.keras.layers.concatenate([layer_8_inception, layer_10_inception, layer_12_inception], axis = 3)

flat_1 = Flatten()(mid_2)

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
    first_five = my_model.layers[: 8]
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