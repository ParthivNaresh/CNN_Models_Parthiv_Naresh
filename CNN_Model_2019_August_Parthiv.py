# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from shutil import move
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import RMSprop
# Import the inception model  
from tensorflow.keras.applications.inception_v3 import InceptionV3

tfe.enable_eager_execution()

zip_data_file = "https://s3-ap-southeast-1.amazonaws.com/he-public-data/DL%23+Beginner.zip"
zip_extract_location = "C:\\Users\\ParthivNaresh\\Documents\\animals_dataset"

external_drive_location = "D:\\CNN_Project_1_Animals_Data"

data_directory = external_drive_location + "\\data\\"
training_directory = external_drive_location + "\\train\\"
testing_directory = external_drive_location + "\\test\\"
predicting_directory = external_drive_location + "\\predict\\"
labels_directory = external_drive_location + "\\animals_labels_train.csv"
inceptionv3_weights = external_drive_location + "\\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
# This is the dataframe that reads from the csv that has all the labels for the image ids
data = pd.read_csv(labels_directory).rename(columns={'Image_id':'image_id','Animal':'animal'})
# Replaced '+' with a space because some labels were two words e.g. "German+Shepherd"
data['animal'] = data.animal.str.replace('\+', ' ')
#print(data[:10])


'''
This data splitting class assumes that the data is initially presented in one folder
and has already been shuffled.
This class splits the data into a training and testing set with a ratio of 80% training
and 20% testing.
Categorization of the data into subfolders based on their labels is done in 
categorize_data().
'''
class split_data_training_test():
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.all_the_data = []
        for image_id in os.listdir(self.data_path):
            if os.path.isfile(self.data_path + image_id):
                self.all_the_data.append(image_id)
        
        self.train_list = train_test_split(self.all_the_data, train_size = 0.8, shuffle=False)[0]
        self.test_list = train_test_split(self.all_the_data, train_size = 0.8, shuffle=False)[1]
        print(str(len(self.train_list)) + " images placed in the training set")
        print(str(len(self.test_list)) + " images placed in the test set")
        #print(self.train_list[0:5])
        #print(self.test_list[0:5])
        
    def move_training_to(self, train_path):
        self.train_path = train_path
        
        for train_name in self.train_list:
            move(self.data_path + train_name,self.train_path)
        
    def move_test_to(self, test_path):
        self.test_path = test_path
        
        for test_name in self.test_list:
            move(self.data_path + test_name,self.test_path)
'''        
print("Splitting data...")
data_split = split_data_training_test(data_directory)
data_split.move_training_to(training_directory)
print("Data split into training directory")
data_split.move_test_to(testing_directory)
print("Data split into test directory")
'''
'''
This categorization class assumes that the data has been presented in the following format:
1 csv file with all the image ids in one column and their corresponding labels in another
2 subfolders (training and testing) that hold a series of images with their file names as their image ids
It will then create a set of folders within the training and testing folders corresponding to all
unique labels from the csv file, and will move the images in the training and testing folders
into their respective subfolders based on how they have been labeled in the csv file.
This will make it easier to use the data in image generators later on.
'''
class categorize_data():
  
  # Initializes the distinct categories in the animal column
  def __init__(self, data):
    self.data = data
    self.animal_categories = list(self.data.animal.unique())
    
  def in_directory(self, directory):
    self.directory = directory
    
    # Makes a sub-directory for every animal in the specified directory
    for animal in self.animal_categories:
        if not os.path.exists(self.directory + animal):
            os.mkdir(self.directory + animal)
    
    # Iterates through every file in the specified directory
    # and finds the same file name in the csv with its
    # relevant animal category and moves it to that animal folder
    for image_id in os.listdir(self.directory):
        if os.path.isfile(self.directory + image_id):
            # Finds the row number that matches the current image-id in the dataframe
            row = self.data.loc[self.data['image_id'] == image_id]
            # iloc is needed to identify the value by INDEX
            animal = row['animal'].iloc[0]
            self.this_file = self.directory + image_id
            self.destination = self.directory + "\\" + animal + "\\"
            # Moves the image to the appropriate animal folder
            move(self.this_file, self.destination)
'''
animal_types = categorize_data(data)
animal_types.in_directory(training_directory)
print("Data categorized in the training directory")
animal_types.in_directory(testing_directory)
print("Data categorized in the testing directory")
'''
'''
This class assumes that all the data has been moved into appropriately labeled
subfolders in the training or testing folders.
This allows for a quick display of random images from that category.
'''
class display():
  
  def __init__(self, animal, directory):
    self.animal = animal
    self.directory = directory
    self.animal_folder = os.path.join(self.directory + self.animal)
    self.number_of_images = len(os.listdir(self.animal_folder))
    print("Total training " + self.animal +  " images: ", self.number_of_images)
    
  def numberOfTimes(self, number):
    # List of image paths in the specified category
    random_list = []
    for index in range(number):
        random_list.append(np.random.randint(0,self.number_of_images))
    
    # Creates a list based on randomly picked images
    self.animal_folder_images = [os.path.join(self.animal_folder, os.listdir(self.animal_folder)[image_index]) 
                                 for image_index in random_list]

    for i, file_path in enumerate(self.animal_folder_images):
      file_name = file_path[file_path.rfind("\\") + 1:]
      image_one = mpimg.imread(file_path)
      plt.imshow(image_one)
      plt.axis('Off')
      plt.title(file_name, loc='center')
      plt.show()

#display("buffalo", training_directory).numberOfTimes(4)


pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                include_top = False, 
                                weights = None  )
pre_trained_model.load_weights(inceptionv3_weights)

# Make all the layers in the pre-trained model non-trainable
for layer in pre_trained_model.layers:
  layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output
  
x = Flatten()(last_output)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(30, activation='softmax')(x)

model = Model(pre_trained_model.input, predictions)

# Compile the model and specify the loss function, optimizer, and metrics to track
model.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'categorical_crossentropy',
              metrics=['acc'])

# Print the model summary
model.summary()

# This callback stops training if the accuracy reaches 90%
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.90):
            print("\nReached 90% accuracy so cancelling training!")
            self.model.stop_training = True
      
callbacks = myCallback()


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
        batch_size = 20,
        target_size=(150,150),
        class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
        testing_directory,
        batch_size = 20,
        target_size=(150,150),
        class_mode='categorical')


# Fit the model using the training and validation generators, specify the number of 
# epochs to train for and how descriptive the output should be
history = model.fit_generator(
        train_generator,
        validation_data = validation_generator,
        steps_per_epoch = 100,
        epochs = 100,
        validation_steps = 50,
        verbose = 1,
        callbacks=[callbacks])


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