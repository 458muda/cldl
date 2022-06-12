from osgeo import gdal
from osgeo import gdal
import sys
import datetime
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import random
import cv2
import pandas as pd
from tensorflow.keras.utils import Sequence
import albumentations as A
import cv2
import sklearn
import segmentation_models
import argparse


from tensorflow.keras import optimizers
#from keras.utils.np_utils import to_categorical
# from tensorflow.keras.utils import multi_gpu_model, plot_model
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

# import tensorflow as tf




# band_path = 'test_data/bands/LC08_L1TP_118065_20180710_20180717_01_T1_band_chip260.tif'
# label_path = 'test_data/labels/LC08_L1TP_118065_20180710_20180717_01_T1_label_chip260.tif'

def read_bands(path):
    band_array = []
    file = gdal.Open(path)
    res_array = np.zeros((256, 256, 6))
    for num in range(6):
        array = np.array(file.GetRasterBand(num+1).ReadAsArray())
        res_array[:,:,num] = array
    band_array.append(res_array)
    del file
    return np.array(band_array).squeeze()

def read_labels(path):
    label_array = []
    file = gdal.Open(path)
    array = np.array(file.GetRasterBand(1).ReadAsArray())
    label_array.append(array)
    del file

    return np.array(label_array).squeeze()

class DataGenerator(Sequence):
  def __init__(self, data_csv_path, batch_size,num_bands=6, transform = None):
    _data = pd.read_csv(data_csv_path)
    self.path_list = _data['bands'].tolist()
    self.labels_list = _data['labels'].tolist()
    self.batch_size = batch_size
    self.transform = transform
    self.num_bands = num_bands
    

  def __len__(self):
    return int(np.floor(len(self.path_list) / self.batch_size))

  def __getitem__(self,index):
    # print('working on batch ',index)
    batch_bands_paths = self.path_list[index*self.batch_size:(index+1)*self.batch_size]
    batch_labels_paths = self.labels_list[index*self.batch_size:(index+1)*self.batch_size]
    

    x = np.empty((self.batch_size, 256, 256, self.num_bands), dtype=np.float32)
    l = np.empty((self.batch_size, 256, 256), dtype=np.float32)

    if self.transform is None:
        for idx, data in enumerate(zip(batch_bands_paths,batch_labels_paths)):
            b_path = data[0]
            l_path = data[1]
            x[idx] = read_bands(b_path)
            l[idx] = read_labels(l_path)
    else:
        for idx, data in enumerate(zip(batch_bands_paths,batch_labels_paths)):
            b_path = data[0]
            l_path = data[1]
            image = read_bands(b_path)
            masks = read_labels(l_path)

            transformed = self.transform(image=image, masks=masks)
            x[idx] = transformed['image']
            l[idx] = transformed['masks']


    return x,l

def binary_unet(input_height, input_width, bandNum):
    inputs = Input((input_height,input_width, bandNum))
    # Block one
    # (256, 256, numBands) -> (256, 256, 64)
    conv1 = BatchNormalization()(Conv2D(32, 3, padding='same', name='Conv1_1', kernel_initializer='he_normal')(inputs))
    conv1 = Activation('relu')(conv1)
    conv1 = BatchNormalization()(Conv2D(32, 3, padding='same', name='Conv1_2', kernel_initializer='he_normal')(conv1))
    conv1 = Activation('relu')(conv1)
    # (256, 256, 64) -> (128, 128, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Block Two
    # (128, 128, 64) -> (128, 128, 128)
    conv2 = BatchNormalization()(Conv2D(64, 3, padding='same', name='Conv2_1', kernel_initializer='he_normal')(pool1))
    conv2 = Activation('relu')(conv2)
    conv2 = BatchNormalization()(Conv2D(64, 3, padding='same', name='Conv2_2', kernel_initializer='he_normal')(conv2))
    conv2 = Activation('relu')(conv2)
    # (128, 128, 128) -> (64, 64, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Block three
    # (64, 64, 128) -> (64, 64, 256)
    conv3 = BatchNormalization()(Conv2D(128, 3, padding='same', name='Conv3_1', kernel_initializer='he_normal')(pool2))
    conv3 = Activation('relu')(conv3)
    conv3 = BatchNormalization()(Conv2D(128, 3, padding='same', name='Conv3_2', kernel_initializer='he_normal')(conv3))
    conv3 = Activation('relu')(conv3)
    # (64, 64, 256) -> ( 32, 32, 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Block four
    # (32, 32, 256) -> (32, 32, 512)
    conv4 = BatchNormalization()(Conv2D(256, 3, padding='same', name='Conv4_1', kernel_initializer='he_normal')(pool3))
    conv4 = Activation('relu')(conv4)
    conv4 = BatchNormalization()(Conv2D(256, 3, padding='same', name='Conv4_2', kernel_initializer='he_normal')(conv4))
    conv4 = Activation('relu')(conv4)
    # (32, 32, 512) -> (16, 16, 512)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Block five
    # (16, 16, 512) -> (16, 16, 1024)
    conv5 = BatchNormalization()(Conv2D(512, 3, padding='same', name='Conv5_1', kernel_initializer='he_normal')(pool4))
    conv5 = Activation('relu')(conv5)
    conv5 = BatchNormalization()(Conv2D(512, 3, padding='same', name='Conv5_2', kernel_initializer='he_normal')(conv5))
    conv5 = Activation('relu')(conv5)
    
    ## Decoder 
    # Block six
    # (16, 16, 1024) -> (32, 32, 1024) -> (32, 32, 512)
    up6 = Conv2DTranspose(256, (4, 4), strides=(2, 2), name = 'Conv6_1', padding='same')(conv5)
    # (32, 32, 512) -> (32, 32, 1024)
    merge6 = concatenate([conv4,up6],axis = 3)   
    # (32, 32, 1024) -> (32, 32, 512)
    conv6 = (Conv2D(256, 3,  padding = 'same', name = 'Conv6_2', kernel_initializer = 'he_normal')(merge6))
    conv6 = Activation('relu')(conv6) 
    conv6 = (Conv2D(256, 3, padding = 'same', name = 'Conv6_3', kernel_initializer = 'he_normal')(conv6))
    conv6 = Activation('relu')(conv6) 
    
    # Block seven 
    # (32, 32, 512) -> (64, 64, 512) -> (64, 64, 256)
    up7 = Conv2DTranspose(128, (4, 4), strides=(2, 2), name = 'Conv7_1', padding='same')(conv6)
    # (64, 64, 256) -> (64, 64, 512)
    merge7 = concatenate([conv3,up7],axis = 3)
    # (64, 64, 512) -> (64, 64, 256)
    conv7 = (Conv2D(128, 3,  padding = 'same', name = 'Conv7_2', kernel_initializer = 'he_normal')(merge7))
    conv7 = Activation('relu')(conv7)          
    conv7 = (Conv2D(128, 3, padding = 'same', name = 'Conv7_3', kernel_initializer = 'he_normal')(conv7))
    conv7 = Activation('relu')(conv7)  
                       
    # Block eight 
    # (64, 64, 256) -> (128, 128, 256) -> (128, 128, 128)
    up8 = Conv2DTranspose(64, (4, 4), strides=(2, 2), name = 'Conv8_1', padding='same')(conv7)
    # (128, 128, 128) -> (128, 128, 256)
    merge8 = concatenate([conv2,up8],axis = 3)   
    # (128, 128, 256) -> (128, 128, 128)
    conv8 = (Conv2D(64, 3, padding = 'same', name = 'Conv8_2', kernel_initializer = 'he_normal')(merge8))
    conv8 = Activation('relu')(conv8)                     
    conv8 = (Conv2D(64, 3,  padding = 'same', name = 'Conv8_3', kernel_initializer = 'he_normal')(conv8))
    conv8 = Activation('relu')(conv8)  
    
    # Block nine 
    # (128, 128, 128) -> (256, 256, 128) -> (256, 256, 64)
    up9 = Conv2DTranspose(32, (4, 4), strides=(2, 2), name = 'Conv9_1', padding='same')(conv8)
    # (256, 256, 64) -> (256, 256, 128)
    merge9 = concatenate([conv1,up9],axis = 3)
    # (256, 256, 128) -> (256, 256, 64)                  
    conv9 = (Conv2D(32, 3, padding = 'same',name = 'Conv9_2', kernel_initializer = 'he_normal')(merge9))
    conv9 = Activation('relu')(conv9)                   
    conv9 = (Conv2D(32, 3, padding = 'same', name = 'Conv9_3', kernel_initializer = 'he_normal')(conv9))
    conv9 = Activation('relu')(conv9)  
    
    # (256, 256, 64) -> (256, 256, 64)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    model = Model(inputs, outputs = conv10)
    
    return model

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str,help='Path to train_csv',required=True)
    parser.add_argument('--val_csv', type=str,help='Path to val_csv',required=False)
    args = parser.parse_args()

    train_data_csv = args.train_csv
    val_data_csv = args.val_csv
    model_1 = binary_unet(256,256,6)
    # print(model_1.summary())
    loss = segmentation_models.losses.BinaryFocalLoss()
    adam_opt = optimizers.Adam(lr=1E-4)
    model_1.compile(loss=loss,optimizer = adam_opt,metrics=['accuracy',segmentation_models.metrics.IOUScore()]) 



    train_data = DataGenerator(train_data_csv,4)
    model_1.fit(train_data,epochs=3)

