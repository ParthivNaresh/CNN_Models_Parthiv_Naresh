# -*- coding: utf-8 -*-

import pandas as pd

zip_data_file = "https://s3-ap-southeast-1.amazonaws.com/he-public-data/DL%23+Beginner.zip"
zip_extract_location = "C:\\Users\\ParthivNaresh\\Documents\\animals_dataset"

external_drive_location = "D:\\CNN_Project_1_Animals_Data"

data_directory = external_drive_location + "\\data\\"
training_directory = external_drive_location + "\\train\\"
testing_directory = external_drive_location + "\\test\\"
predicting_directory = external_drive_location + "\\predict\\"
labels_directory = external_drive_location + "\\animals_labels_train.csv"
inceptionv3_weights = external_drive_location + "\\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

data = pd.read_csv(labels_directory).rename(columns={'Image_id':'image_id','Animal':'animal'})
data['animal'] = data.animal.str.replace('\+', ' ')