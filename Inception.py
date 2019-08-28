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
layer_2 = Conv2D(32, (3,3), activation='relu')(layer_1)
layer_3 = MaxPooling2D((2,2))(layer_2)

layer_4 = Conv2D(64, (3,3), activation='relu')(layer_3)
layer_5 = MaxPooling2D((2,2))(layer_4)
    
### 1st layer
layer_1_1_inception = Conv2D(20, (1,1), padding='same', activation='relu')(layer_5)
layer_1_2_inception = MaxPooling2D((2,2), strides=(1,1), padding='same')(layer_5)
layer_1_3_inception = Conv2D(12, (1,1), padding='same', activation='relu')(layer_5)
layer_1_4_inception = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_5)
layer_1_5_inception = Conv2D(12, (1,1), padding='same', activation='relu')(layer_5)

### 2nd layer
layer_2_1_inception = Conv2D(12, (1,3), padding='same', activation='relu')(layer_1_1_inception)
layer_2_2_inception = Conv2D(24, (1,1), padding='same', activation='relu')(layer_1_2_inception)
layer_2_3_inception = Conv2D(12, (1,5), padding='same', activation='relu')(layer_1_3_inception)
layer_2_4_inception = Conv2D(24, (1,1), padding='same', activation='relu')(layer_1_4_inception)
layer_2_5_inception = Conv2D(12, (1,3), padding='same', activation='relu')(layer_1_5_inception)

### 3rd layer
layer_3_1_inception = Conv2D(12, (3,1), padding='same', activation='relu')(layer_2_1_inception)
layer_3_2_inception = Conv2D(12, (5,1), padding='same', activation='relu')(layer_2_3_inception)
layer_3_3_inception = Conv2D(12, (3,1), padding='same', activation='relu')(layer_2_5_inception)

mid_1 = tf.keras.layers.concatenate([layer_3_1_inception, layer_2_2_inception,
                                     layer_3_2_inception, layer_2_4_inception,
                                     layer_3_3_inception], axis = 3)

layer_6 = Conv2D(84, (3,3), activation='relu')(mid_1)
layer_7 = MaxPooling2D((2,2))(layer_6)

layer_8 = Conv2D(96, (3,3), activation='relu')(layer_7)
layer_9 = MaxPooling2D((2,2))(layer_8)

### 4st layer
layer_4_1_inception = Conv2D(24, (1,1), padding='same', activation='relu')(layer_9)
layer_4_2_inception = MaxPooling2D((2,2), strides=(1,1), padding='same')(layer_9)
layer_4_3_inception = Conv2D(24, (1,1), padding='same', activation='relu')(layer_9)

### 5th layer
layer_5_1_inception = Conv2D(24, (3,3), padding='same', activation='relu')(layer_4_1_inception)
layer_5_2_inception = Conv2D(16, (1,1), padding='same', activation='relu')(layer_4_2_inception)
layer_5_3_inception = Conv2D(24, (1,3), padding='same', activation='relu')(layer_4_3_inception)
layer_5_4_inception = Conv2D(24, (3,1), padding='same', activation='relu')(layer_4_3_inception)

### 5th layer
layer_6_1_inception = Conv2D(24, (1,3), padding='same', activation='relu')(layer_5_1_inception)
layer_6_2_inception = Conv2D(24, (3,1), padding='same', activation='relu')(layer_5_1_inception)

mid_2 = tf.keras.layers.concatenate([layer_5_2_inception, layer_5_3_inception,
                                     layer_5_4_inception, layer_6_1_inception,
                                     layer_6_2_inception], axis = 3)

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