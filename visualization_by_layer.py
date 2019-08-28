# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np
import random
from   tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
from Inception import my_model
from constants import training_image, training_directory, visualizations
from PIL import Image

np.seterr(divide='ignore', invalid='ignore')

class visualization_by_layer():
    
    def __init__(self, animal):
        # Creates a list of every animal in the specified folder
        self.path = training_directory + animal
        self.all_the_data = []
        for image_id in os.listdir(self.path):
            if os.path.isfile(self.path + "\\" + image_id):
                self.all_the_data.append(image_id)
        # Randomly chooses an animal in the aggregated file paths
        img_path = random.choice(self.all_the_data)
        img = load_img(self.path + "\\" + img_path, target_size=(150, 150))
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        x /= 255.0
        
        # Avoids the input layer as part of the outputs
        successive_layers = [layer for layer in my_model.layers[0:] if not layer.name.startswith('input')]
        successive_outputs = [layer.output for layer in successive_layers]
        visualization_model = tf.keras.models.Model(inputs = my_model.input, outputs = successive_outputs)
        successive_feature_maps = visualization_model.predict(x)

        position = 0
        list_of_layers = []
        for layer, feature_map in zip(successive_layers, successive_feature_maps):

            if len(feature_map.shape) == 4:
                
                # Only for the conv/maxpool layers, not the fully-connected layers
                # feature map shape (1, size, size, n_features)
                n_features = feature_map.shape[-1]
                size = feature_map.shape[1]
            
                # Tile images in this matrix
                display_grid = np.zeros((size, size * n_features))
                
                # Postprocess the feature
                for i in range(n_features):
                    x  = feature_map[0, :, :, i]
                    x -= x.mean()
                    x /= x.std()
                    x *=  64
                    x += 128
                    x  = np.clip(x, 0, 255).astype('uint8')
                    # Tile each filter into a horizontal grid
                    display_grid[:, i * size : (i + 1) * size] = x
                
                width = 20. / n_features
                height = 10. / n_features
                plt.rcParams.update({'figure.max_open_warning': 0})
                plt.figure(figsize=(width * n_features, height))
                
                # Function that formats the title of the layer and provides
                # cursory information based on the type of layer
                def get_layer_information(layer_name):
                    
                    information = []
                    title = ''
                    
                    layers_names = {
                            'conv2d' : "Convolutional 2D",
                            'conv3d' : "Convolutional 3D",
                            'max_pooling' : "Max Pooling",
                            'concatenate' : "Concatenate"
                            }
                    
                    layers_properties = {
                            'conv2d' : ['filters', 'kernel_size', 'strides'],
                            'max_pooling' : ['pool_size', 'strides'],
                            'concatenate' : ['axis']
                            }
                    
                    for key in layers_names:
                        if layer_name.startswith(key):
                            name = layers_names.get(key, "nothing")
                            information.append(name)
                            title += name
                            break;
                    
                    for key in layers_properties:
                        if layer_name.startswith(key):
                            prop = layers_properties.get(key, "nothing")
                            for prop_type in prop:
                                information.append(str(layer.get_config()[prop_type]))
                            break;
                            
                    if key == 'conv2d':
                        plt.title(title + ", Filters: " + information[1] 
                                  + ", Kernel Size: " + information[2]
                                  + ", Strides: " + information[3])
                    elif key == 'max_pooling':
                        plt.title(title + ", Pool Size: " + information[1] 
                                  + ", Strides: " + information[2])
                    elif key == 'concatenate':
                        plt.title(title + ", Axis: " + information[1])
                
                get_layer_information(layer.name.lower())
                
                plt.grid(False)
                # Displays the plot in the console
                plt.imshow(display_grid, aspect='auto', cmap='viridis')
                # Saves the output of every layer in the specified folder so they
                # can be combined later
                plt.savefig(visualizations + '\\' + str(position) + '.png', bbox_inches='tight')
                list_of_layers.append(visualizations + '\\' + str(position) + '.png')
                
                position += 1

        images = []
        for each_layer in list_of_layers:
            images.append(each_layer)
        
        widths, heights = zip(*(Image.open(each_image).size for each_image in images))
        # Because of line 63, the width will be more or less the same for
        # every output (depending on # of filters), however some are wider 
        # than others by a few pixels leading to black space behind the image,
        # so the minimum width is taken instead of the maximum
        min_width = min(widths)
        total_height = sum(heights)
        
        new_im = Image.new('RGB', (min_width, total_height))
        
        y_offset = 0    
        for im in images:
          new_im.paste(Image.open(im), (0,y_offset))
          y_offset += Image.open(im).size[1]
        
        # Removes the files that each contain the output of one layer
        for each_file in os.listdir(visualizations):
            if os.path.isfile(visualizations + '\\' + each_file):
                os.remove(visualizations + '\\' + each_file)
            
        new_im.save(visualizations + '\\output_by_layers.jpg')
            
                
visualization_by_layer("wolf")