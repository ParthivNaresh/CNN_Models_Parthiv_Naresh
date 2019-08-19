# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

'''
This class assumes that all the data has been moved into appropriately labeled
subfolders in the training or testing folders.
This allows for a quick display of random images from that category.
'''
class display_images():
  
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


