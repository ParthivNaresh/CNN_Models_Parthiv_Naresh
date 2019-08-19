# -*- coding: utf-8 -*-

import os
from shutil import move

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
