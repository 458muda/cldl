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
import math


from tensorflow.keras import optimizers
#from keras.utils.np_utils import to_categorical
# from tensorflow.keras.utils import multi_gpu_model, plot_model
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K



import tensorflow as tf




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
            image = image.astype(np.float32)
            mask = read_labels(l_path)
            mask = mask.astype(np.float32)

            transformed = self.transform(image=image, mask=mask)
            
            x[idx] = transformed['image']
            l[idx] = transformed['mask']


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
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
])

def numpy_isqrt(number):
    threehalfs = 1.5
    x2 = number * 0.5
    y = np.float32(number)
    
    i = y.view(np.int32)
    i = np.int32(0x5f3759df) - np.int32(i >> 1)
    y = i.view(np.float32)
    
    y = y * (threehalfs - (x2 * y * y))
    return float(y)


def scheduler(epoch, lr):
    d_model = 100000
    d_model = tf.cast(d_model, tf.float32)
    warmup_steps = 50

    arg1 = numpy_isqrt(epoch)
    arg2 = epoch * (warmup_steps ** -1.5)

    return numpy_isqrt(d_model) * min(arg1, arg2)



class LRTensorBoard(TensorBoard):
    # add other arguments to __init__ if you need
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str,help='Path to train_csv',required=True)
    parser.add_argument('--val_csv', type=str,help='Path to val_csv',required=False)
    parser.add_argument('--log_dir', type=str,help='Path to log dir',required=True)

    args = parser.parse_args()

    train_data_csv = args.train_csv
    val_data_csv = args.val_csv
    logdir = args.log_dir
    
    os.environ["CUDA_VISIBLE_DEVICES"]= "0"
    print(tf.test.is_gpu_available())
    print(tf.__version__)
    
    loss = segmentation_models.losses.BinaryFocalLoss()
    
    #d_model = 0.1
    #learning_rate = CustomSchedule(d_model)
    adam_opt = optimizers.Adam()
    #adam_opt = optimizers.Adam(lr=1E-4)
    
    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

     
    tensorboard_callback = LRTensorBoard(logdir, histogram_freq=1)
    early_stop_callback = EarlyStopping(monitor = 'val_loss', patience = 20, verbose=1)
    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
     #                         patience=10, min_lr=0.00001, verbose=1)
    train_data = DataGenerator(train_data_csv,64,transform=transform)
    val_data = DataGenerator(val_data_csv,64)
    
    
    

## two GPUs, so the devices are gpu0, and gpu1. If you try more GPUs, remember to change the devices.
    #strategy = tf.distribute.MirroredStrategy(devices= ["/gpu:0","/gpu:1","/gpu:2","/gpu:3"],cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    #print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    
    #with strategy.scope():
    model_1 = binary_unet(256,256,6)
    # print(model_1.summary())

    model_1.compile(loss=loss,optimizer = adam_opt,metrics=['accuracy',segmentation_models.metrics.IOUScore()])


    model_1.fit(train_data,validation_data=val_data,epochs=500, callbacks=[tensorboard_callback, early_stop_callback, lr_callback])
    
    model_path = os.path.join("/home/vishal/DL_results", "unet64_lrc_500000_100"+".h5")
    model_1.save(model_path)
    
    
    

