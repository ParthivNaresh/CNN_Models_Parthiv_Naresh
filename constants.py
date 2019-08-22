# -*- coding: utf-8 -*-

import pandas as pd

zip_data_file = "https://s3-ap-southeast-1.amazonaws.com/he-public-data/DL%23+Beginner.zip"
zip_extract_location = "C:\\Users\\ParthivNaresh\\Documents\\animals_dataset"

external_drive_location = "D:\\CNN_Project_1_Animals_Data"
github_drive_location = "D:\\CNN_Project_1_Animals_Of_Africa"

data_directory = external_drive_location + "\\data\\"
training_directory = external_drive_location + "\\train\\"
testing_directory = external_drive_location + "\\test\\"
predicting_directory = external_drive_location + "\\predict\\"
labels_directory = external_drive_location + "\\animals_labels_train.csv"

training_image = "D:\\train\\"

callback_cutoff_accuracy = 0.6

data = pd.read_csv(labels_directory).rename(columns={'Image_id':'image_id','Animal':'animal'})
data['animal'] = data.animal.str.replace('\+', ' ')