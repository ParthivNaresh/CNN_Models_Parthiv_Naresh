# -*- coding: utf-8 -*-

import os
from shutil import move
from sklearn.model_selection import train_test_split

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
