# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np
import random
from   tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
from Inception import my_model
from constants import training_directory

class visualization_by_layer():
    
    def __init__(self, animal):
        self.path = training_directory + animal
        self.all_the_data = []
        for image_id in os.listdir(self.path):
            if os.path.isfile(self.path + "\\" + image_id):
                self.all_the_data.append(image_id)
        
        img_path = random.choice(self.all_the_data)
        img = load_img(self.path + "\\" + img_path, target_size=(150, 150))
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        x /= 255.0
        
        # Avoids the input layer as part of the outputs
        successive_outputs = [layer.output for layer in my_model.layers[0:] if not layer.name.startswith('input')]
        visualization_model = tf.keras.models.Model(inputs = my_model.input, outputs = successive_outputs)
        successive_feature_maps = visualization_model.predict(x)

        layer_names = [layer.name for layer in successive_outputs]
        
        for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  
            if len(feature_map.shape) == 4:
                # Only for the conv/maxpool layers, not the fully-connected layers
                # feature map shape (1, size, size, n_features)
                n_features = feature_map.shape[-1]
                size = feature_map.shape[1]
            
                # Tile our images in this matrix
                display_grid = np.zeros((size, size * n_features))
                
                # Postprocess the feature to be visually palatable
                for i in range(n_features):
                    x  = feature_map[0, :, :, i]
                    x -= x.mean()
                    x /= x.std ()
                    x *=  64
                    x += 128
                    x  = np.clip(x, 0, 255).astype('uint8')
                    display_grid[:, i * size : (i + 1) * size] = x # Tile each filter into a horizontal grid
                
                scale = 20. / n_features
                plt.figure( figsize=(scale * n_features, scale) )
                plt.title (layer_name)
                plt.grid  (False)
                plt.imshow(display_grid, aspect='auto', cmap='viridis')
                
a = visualization_by_layer("Lion")