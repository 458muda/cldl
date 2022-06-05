from osgeo import gdal
from osgeo import gdal
import sys
import tensorflow as tf
import datetime
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import random
from tensorflow.keras import optimizers
#from keras.utils.np_utils import to_categorical
#from tensorflow.keras.utils import multi_gpu_model, plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, CSVLogger
from tensorflow.keras import backend as K
import cv2
import sklearn
import segmentation_models

def scene_folder_list(catagory, pathrow_list, data_source_folder='/workspace/_libs/DL_Training_Library'):
    """
This function is designed to return the path of chips folders the user requests
    :param catagoryr: the catagory of landcover. We have four catagories: Mangrove, Water, Wetland, and Pond
    :param pathrow_list: a list of pathrow IDs.eg: ['128051','011060']
    :param data_source_folder: the data source folder of chips. On the blade, it is the default path. On Sandy, it should be r'//sandy.local/projects/Moore_Automation/DL_Training_Library'
    :return: the list of paths of chips folders.
    """
    source_folder = os.path.join(data_source_folder,catagory)
    res_folder_list = []
    total_num = len(pathrow_list)
    success_num = 0
    for pathrow in pathrow_list:
        for folder in os.listdir(source_folder):
            if folder.endswith(pathrow):
                res_folder_list.append(os.path.join(source_folder,folder))
                success_num+=1
    print(str(success_num),'scenes exist for',str(total_num),'required scenes')
    return res_folder_list

def read_scenes(scene_folder_list):
    """
This function is to read bands and label from a list of scene folder paths. Without random order.
    :param scene_folder_list: a list of chip folder paths
    :return: numpy arrays of bands and the label
    """
    for i in range(len(scene_folder_list)):
        if(i==0):
            folder = scene_folder_list[i]
            res_bands, res_label = read_scene_band_label(folder)
        else:
            folder = scene_folder_list[i]
            scene_bands, scene_label = read_scene_band_label(folder)
            res_bands = np.append(res_bands, scene_bands, axis=0)
            res_label = np.append(res_label, scene_label, axis=0)
    return res_bands, res_label


def read_scene_band_label(scene_folder):
    """
This function is to read bands and label from the scene folder
    :param scene_folder: the folder path of a scene
    :return: numpy arrays of bands and the label
    """
    # empty list to save the band and label
    band_array = []
    label_array = []
    # get the band and label folder
    for folder in os.listdir(scene_folder):
        if(folder.endswith('bands')):
            scene_band_folder = os.path.join(scene_folder,folder)
        if(folder.endswith('label')):
            scene_label_folder = os.path.join(scene_folder,folder)
    # save each band chips to the list
    band_list = os.listdir(scene_band_folder)
    label_list = os.listdir(scene_label_folder)
    band_list.sort()
    label_list.sort()
    for i in band_list:
        file = gdal.Open(os.path.join(scene_band_folder,i))
        res_array = np.zeros((256, 256, 6))
        for num in range(6):
            array = np.array(file.GetRasterBand(num+1).ReadAsArray())
            res_array[:,:,num] = array
        band_array.append(res_array)
        del file
    # save each label chip to the list
    for i in label_list:
        file = gdal.Open(os.path.join(scene_label_folder,i))
        array = np.array(file.GetRasterBand(1).ReadAsArray())
        label_array.append(array)
        del file
    return np.array(band_array), np.array(label_array)


# rotation for 90, 180, 270, and horizotal and vertical flip
def clarkAug(bands, label):
    ## rotation 1 for 90, 2 for 180, 3 for 270
    image_rot90 = np.rot90(bands, 1, axes=(1, 2))
    label_rot90 = np.rot90(label, 1, axes=(1, 2))
    image_rot180 = np.rot90(bands, 2, axes=(1, 2))
    label_rot180 = np.rot90(label, 2, axes=(1, 2))
    image_rot270 = np.rot90(bands, 3, axes=(1, 2))
    label_rot270 = np.rot90(label, 3, axes=(1, 2))
    # axis=1 vertical flip, axis=2 horizontasl flip
    image_vflip = np.flip(bands, axis=1)
    label_vflip = np.flip(label, axis=1)
    image_hflip = np.flip(bands, axis=2)
    label_hflip = np.flip(label, axis=2)
    res_bands = np.vstack([bands, image_rot90, image_rot180, image_rot270, image_vflip, image_hflip])
    res_label = np.vstack([label, label_rot90, label_rot180, label_rot270, label_vflip, label_hflip])
    np.random.seed(42)
    np.random.shuffle(res_bands)
    np.random.seed(42)
    np.random.shuffle(res_label)
    return res_bands, res_label


### Import the important libraries
from osgeo import gdal
import tensorflow as tf
import datetime
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import random
from tensorflow.keras import optimizers
#from keras.utils.np_utils import to_categorical
#from tensorflow.keras.utils import multi_gpu_model, plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";


def binary_unet(input_height, input_width, bandNum):
    inputs = Input((input_height,input_width, bandNum))
    # Block one
    # (256, 256, numBands) -> (256, 256, 64)
    conv1 = BatchNormalization()(Conv2D(64, 3, padding='same', name='Conv1_1', kernel_initializer='he_normal')(inputs))
    conv1 = Activation('relu')(conv1)
    conv1 = BatchNormalization()(Conv2D(64, 3, padding='same', name='Conv1_2', kernel_initializer='he_normal')(conv1))
    conv1 = Activation('relu')(conv1)
    # (256, 256, 64) -> (128, 128, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Block Two
    # (128, 128, 64) -> (128, 128, 128)
    conv2 = BatchNormalization()(Conv2D(128, 3, padding='same', name='Conv2_1', kernel_initializer='he_normal')(pool1))
    conv2 = Activation('relu')(conv2)
    conv2 = BatchNormalization()(Conv2D(128, 3, padding='same', name='Conv2_2', kernel_initializer='he_normal')(conv2))
    conv2 = Activation('relu')(conv2)
    # (128, 128, 128) -> (64, 64, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Block three
    # (64, 64, 128) -> (64, 64, 256)
    conv3 = BatchNormalization()(Conv2D(256, 3, padding='same', name='Conv3_1', kernel_initializer='he_normal')(pool2))
    conv3 = Activation('relu')(conv3)
    conv3 = BatchNormalization()(Conv2D(256, 3, padding='same', name='Conv3_2', kernel_initializer='he_normal')(conv3))
    conv3 = Activation('relu')(conv3)
    # (64, 64, 256) -> ( 32, 32, 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Block four
    # (32, 32, 256) -> (32, 32, 512)
    conv4 = BatchNormalization()(Conv2D(512, 3, padding='same', name='Conv4_1', kernel_initializer='he_normal')(pool3))
    conv4 = Activation('relu')(conv4)
    conv4 = BatchNormalization()(Conv2D(512, 3, padding='same', name='Conv4_2', kernel_initializer='he_normal')(conv4))
    conv4 = Activation('relu')(conv4)
    # (32, 32, 512) -> (16, 16, 512)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Block five
    # (16, 16, 512) -> (16, 16, 1024)
    conv5 = BatchNormalization()(Conv2D(1024, 3, padding='same', name='Conv5_1', kernel_initializer='he_normal')(pool4))
    conv5 = Activation('relu')(conv5)
    conv5 = BatchNormalization()(Conv2D(1024, 3, padding='same', name='Conv5_2', kernel_initializer='he_normal')(conv5))
    conv5 = Activation('relu')(conv5)
    
    ## Decoder 
    # Block six
    # (16, 16, 1024) -> (32, 32, 1024) -> (32, 32, 512)
    up6 = Conv2DTranspose(512, (4, 4), strides=(2, 2), name = 'Conv6_1', padding='same')(conv5)
    # (32, 32, 512) -> (32, 32, 1024)
    merge6 = concatenate([conv4,up6],axis = 3)   
    # (32, 32, 1024) -> (32, 32, 512)
    conv6 = (Conv2D(512, 3,  padding = 'same', name = 'Conv6_2', kernel_initializer = 'he_normal')(merge6))
    conv6 = Activation('relu')(conv6) 
    conv6 = (Conv2D(512, 3, padding = 'same', name = 'Conv6_3', kernel_initializer = 'he_normal')(conv6))
    conv6 = Activation('relu')(conv6) 
    
    # Block seven 
    # (32, 32, 512) -> (64, 64, 512) -> (64, 64, 256)
    up7 = Conv2DTranspose(256, (4, 4), strides=(2, 2), name = 'Conv7_1', padding='same')(conv6)
    # (64, 64, 256) -> (64, 64, 512)
    merge7 = concatenate([conv3,up7],axis = 3)
    # (64, 64, 512) -> (64, 64, 256)
    conv7 = (Conv2D(256, 3,  padding = 'same', name = 'Conv7_2', kernel_initializer = 'he_normal')(merge7))
    conv7 = Activation('relu')(conv7)          
    conv7 = (Conv2D(256, 3, padding = 'same', name = 'Conv7_3', kernel_initializer = 'he_normal')(conv7))
    conv7 = Activation('relu')(conv7)  
                       
    # Block eight 
    # (64, 64, 256) -> (128, 128, 256) -> (128, 128, 128)
    up8 = Conv2DTranspose(128, (4, 4), strides=(2, 2), name = 'Conv8_1', padding='same')(conv7)
    # (128, 128, 128) -> (128, 128, 256)
    merge8 = concatenate([conv2,up8],axis = 3)   
    # (128, 128, 256) -> (128, 128, 128)
    conv8 = (Conv2D(128, 3, padding = 'same', name = 'Conv8_2', kernel_initializer = 'he_normal')(merge8))
    conv8 = Activation('relu')(conv8)                     
    conv8 = (Conv2D(128, 3,  padding = 'same', name = 'Conv8_3', kernel_initializer = 'he_normal')(conv8))
    conv8 = Activation('relu')(conv8)  
    
    # Block nine 
    # (128, 128, 128) -> (256, 256, 128) -> (256, 256, 64)
    up9 = Conv2DTranspose(64, (4, 4), strides=(2, 2), name = 'Conv9_1', padding='same')(conv8)
    # (256, 256, 64) -> (256, 256, 128)
    merge9 = concatenate([conv1,up9],axis = 3)
    # (256, 256, 128) -> (256, 256, 64)                  
    conv9 = (Conv2D(64, 3, padding = 'same',name = 'Conv9_2', kernel_initializer = 'he_normal')(merge9))
    conv9 = Activation('relu')(conv9)                   
    conv9 = (Conv2D(64, 3, padding = 'same', name = 'Conv9_3', kernel_initializer = 'he_normal')(conv9))
    conv9 = Activation('relu')(conv9)  
    
    # (256, 256, 64) -> (256, 256, 64)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    model = Model(inputs, outputs = conv10)
    
    return model
    


### this function is to read the data and the relevant properties
def ReadData_geoinf(path):
    """
    This function is used to read geoinformation

    param path: img path

    """
    ds = gdal.Open(path, 0)
    if ds is None:
        sys.exit('Could not open {0}.'.format(path))

    geoTransform = ds.GetGeoTransform()
    proj = ds.GetProjection()

    XSize = ds.RasterXSize
    YSize = ds.RasterYSize
    MinX = geoTransform[0]
    MaxY = geoTransform[3]
    MaxX = MinX + geoTransform[1] * XSize
    MinY = MaxY + geoTransform[5] * YSize

    resolution = geoTransform[1]

    data = ds.ReadAsArray()
    res = {'data': data,
           'geoTransform': geoTransform,
           'projection': proj,
           'minX': MinX,
           'maxX': MaxX,
           'minY': MinY,
           'maxY': MaxY,
           'Xsize': XSize,
           'Ysize': YSize,
           'resolution': resolution}
    return res


## This function is to cut np.ndarray into chips.
def cut_array(data, row, col, row_buffer, col_buffer):
    ### calculate the right low corner index for row and col
    ### (row, col)
    ### the index for numpy array will be [a, b)
    data_row = data.shape[0]
    data_col = data.shape[1]
    if ((data_row - row) % (row - row_buffer) == 0):
        row_list = list(range(row, data_row + 1, row - row_buffer))
    else:
        row_list = list(range(row, data_row + 1, row - row_buffer))
        row_list.append(data_row)
    if ((data_col - col) % (col - col_buffer) == 0):
        col_list = list(range(col, data_col + 1, col - col_buffer))
    else:
        col_list = list(range(col, data_col + 1, col - col_buffer))
        col_list.append(data_col)
    res = []
    for j in col_list:
        for i in row_list:
            res.append(data[i - row:i, j - col:j])

    return np.array(res)


### index is from [0, row_num*col_num)
### return (minX, maxX, minY, maxY)
### This function is to return the coordinate range for the index
def chip_index_finder(index, row, col, data_row, data_col, row_buffer, col_buffer):
    if((data_row-row)%(row-row_buffer)>0):
        row_num = int((data_row-row)/(row-row_buffer))+2
    else:
        row_num = int((data_row-row)/(row-row_buffer))+1
    if((data_col-col)%(col-col_buffer)>0):
        col_num = int((data_col-col)/(col-col_buffer))+2
    else:
        col_num = int((data_col-col)/(col-col_buffer))+1
    row_index = index%row_num
    col_index = int(index/row_num)
    if(row_index == row_num-1):
        row_coor = data_row
    elif(row_index == 0):
        row_coor = row
    else:
        row_coor = row+(row_index)*(row-row_buffer)
    if(col_index == col_num-1):
        col_coor = data_col
    elif(col_index == 0):
        col_coor = col
    else:
        col_coor = col + col_index*(col-col_buffer)
    return (row_coor - row, row_coor, col_coor - col, col_coor)



## this function is to mosaic the chips into a whole image.
## we always use the "half buffer" as the weights
def mosaic_chips(data_array, index_list, weight_array, data_row, data_col, row, col, row_buffer, col_buffer):
    res = np.zeros((data_row, data_col))
    for i in range(len(index_list)):
        chip_index = index_list[i]
        coors = chip_index_finder(chip_index, row, col, data_row, data_col, row_buffer, col_buffer)
        temp_array = weight_array[i,:,:]*data_array[i,:,:]
        res[coors[0]:coors[1],coors[2]:coors[3]] = res[coors[0]:coors[1],coors[2]:coors[3]]+temp_array
    return res


def weights_generator(weight_type, data_row, data_col, row, col, row_buffer, col_buffer):
    if ((data_row - row) % (row - row_buffer) > 0):
        row_num = int((data_row - row) / (row - row_buffer)) + 2
    else:
        row_num = int((data_row - row) / (row - row_buffer)) + 1
    if ((data_col - col) % (col - col_buffer) > 0):
        col_num = int((data_col - col) / (col - col_buffer)) + 2
    else:
        col_num = int((data_col - col) / (col - col_buffer)) + 1
    if (weight_type == 'no_buffer'):
        weight_array = [np.ones((row, col)) for _ in range(row_num * col_num)]
        right_margin = np.zeros((row, col))
        right_cut = (row if (data_row % row == 0) else data_row % row)
        right_margin[row - right_cut:, :] = 1
        down_margin = np.zeros((row, col))
        down_cut = (col if (data_col % col == 0) else data_col % col)
        down_margin[:, col - down_cut:] = 1
        corner_margin = right_margin * down_margin
        for i in range(row_num - 1, row_num * col_num, row_num):
            weight_array[i] = right_margin
        for i in range(row_num * col_num - row_num, row_num * col_num):
            weight_array[i] = down_margin
        weight_array[row_num * col_num - 1] = corner_margin
        weight_array = np.array(weight_array)
    if (weight_type == 'buffer_average'):
        weight_array = []
        template_zeros = np.zeros((data_row, data_col))
        for i in range(col_num * row_num):
            coors = chip_index_finder(i, row, col, data_row, data_col, row_buffer, col_buffer)
            temp_array = np.ones((row, col))
            template_zeros[coors[0]:coors[1], coors[2]:coors[3]] = template_zeros[coors[0]:coors[1],
                                                                   coors[2]:coors[3]] + temp_array
        template_zeros = 1.0 / template_zeros
        for i in range(col_num * row_num):
            coors = chip_index_finder(i, row, col, data_row, data_col, row_buffer, col_buffer)
            weight_array.append(template_zeros[coors[0]:coors[1], coors[2]:coors[3]])
        weight_array = np.array(weight_array)
    if (weight_type == 'buffer_gauss_average'):
        def gaus2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
            return 1. / (2. * np.pi * sx * sy) * np.exp(
                -((x - mx) ** 2. / (2. * sx ** 2.) + (y - my) ** 2. / (2. * sy ** 2.)))

        weight_array = []
        x = np.linspace(-5, 5, row)
        y = np.linspace(-5, 5, col)
        x, y = np.meshgrid(x, y)  # get 2D variables instead of 1D
        template_weights = gaus2d(x, y)
        weight_array = []
        template_zeros = np.zeros((data_row, data_col))
        for i in range(col_num * row_num):
            coors = chip_index_finder(i, row, col, data_row, data_col, row_buffer, col_buffer)
            template_zeros[coors[0]:coors[1], coors[2]:coors[3]] = template_zeros[coors[0]:coors[1],
                                                                   coors[2]:coors[3]] + template_weights

        for i in range(col_num * row_num):
            coors = chip_index_finder(i, row, col, data_row, data_col, row_buffer, col_buffer)
            weight_array.append(template_weights / template_zeros[coors[0]:coors[1], coors[2]:coors[3]])
        weight_array = np.array(weight_array)
    if (weight_type == 'buffer_linear_average'):
        assert row_buffer == col_buffer
        template_weights = np.ones((row, col))
        for i in range(row_buffer):
            pixel_int = 1. / row_buffer
            template_weights[i, i:(col - i)] = pixel_int * i
            template_weights[i:(row - i), i] = pixel_int * i
            template_weights[row - i - 1, i:(col - i)] = pixel_int * i
            template_weights[i:(row - i), col - i - 1] = pixel_int * i
        weight_array = []
        template_zeros = np.zeros((data_row, data_col))
        for i in range(col_num * row_num):
            coors = chip_index_finder(i, row, col, data_row, data_col, row_buffer, col_buffer)
            template_zeros[coors[0]:coors[1], coors[2]:coors[3]] = template_zeros[coors[0]:coors[1],
                                                                   coors[2]:coors[3]] + template_weights

        for i in range(col_num * row_num):
            coors = chip_index_finder(i, row, col, data_row, data_col, row_buffer, col_buffer)
            weight_array.append(template_weights / template_zeros[coors[0]:coors[1], coors[2]:coors[3]])
        weight_array = np.array(weight_array)

    if (weight_type == 'half_buffer'):
        half_row_buffer = int(row_buffer / 2)
        half_col_buffer = int(col_buffer / 2)
        template_weights = np.zeros((row, col))
        template_weights[half_row_buffer:(row - half_row_buffer), half_col_buffer:(col - half_col_buffer)] = 1
        weight_array = [template_weights for _ in range(row_num * col_num)]

        up_border = np.zeros((row, col))
        up_border[half_row_buffer:(row - half_row_buffer), 0:(col - half_col_buffer)] = 1
        down_border = np.zeros((row, col))
        down_border[half_row_buffer:(row - half_row_buffer), half_col_buffer:] = 1
        left_border = np.zeros((row, col))
        left_border[0:(row - half_row_buffer), half_col_buffer:(col - half_col_buffer)] = 1
        right_border = np.zeros((row, col))
        right_border[half_row_buffer:, half_col_buffer:(col - half_col_buffer)] = 1

        up_left_border = np.zeros((row, col))
        up_left_border[0:(row - half_row_buffer), 0:(col - half_col_buffer)] = 1

        up_right_border = np.zeros((row, col))
        up_right_border[half_row_buffer:, 0:(col - half_col_buffer)] = 1
        down_left_border = np.zeros((row, col))
        down_left_border[0:(row - half_row_buffer), half_col_buffer:] = 1
        down_right_border = np.zeros((row, col))
        down_right_border[half_row_buffer:, half_col_buffer:] = 1

        for i in range(0, row_num):
            weight_array[i] = up_border
        for i in range(0, row_num * col_num - 1, row_num):
            weight_array[i] = left_border

        right_cut = ((row - row_buffer) if ((data_row - row) % (row - row_buffer) == 0) else (data_row - row) % (
                    row - row_buffer))
        down_cut = ((col - col_buffer) if ((data_col - col) % (col - col_buffer) == 0) else (data_col - col) % (
                    col - col_buffer))
        #         half_row_buffer:(row - half_row_buffer), half_col_buffer:(col - half_col_buffer)
        if (right_cut == (row - row_buffer)):
            for i in range(row_num - 1, row_num * col_num - 1, row_num):
                weight_array[i] = right_border
            if (down_cut == (col - col_buffer)):
                for i in range(row_num * col_num - row_num, row_num * col_num - 1):
                    weight_array[i] = down_border
                weight_array[0] = up_left_border
                weight_array[row_num - 1] = up_right_border
                weight_array[row_num * (col_num - 1)] = down_left_border
                weight_array[row_num * col_num - 1] = down_right_border
            else:
                down_margin = np.zeros((row, col))
                down_margin[half_row_buffer:(row - half_row_buffer), col - down_cut:] = 1
                left_down_margin = np.zeros((row, col))
                left_down_margin[0:(row - half_row_buffer), col - down_cut:] = 1
                right_down_margin_2 = np.zeros((row, col))
                right_down_margin_2[half_row_buffer:(row), col - down_cut:] = 1
                for i in range(row_num * col_num - row_num, row_num * col_num - 1):
                    weight_array[i] = down_margin
                    weight_array[i - row_num] = down_border
                weight_array[0] = up_left_border
                weight_array[row_num - 1] = up_right_border
                weight_array[row_num * (col_num - 2)] = down_left_border
                weight_array[row_num * (col_num - 1)] = left_down_margin
                weight_array[row_num * col_num - row_num - 1] = down_right_border
                weight_array[row_num * col_num - 2] = right_down_margin_2
        else:
            right_margin = np.zeros((row, col))
            right_margin[row - right_cut:, half_col_buffer:(col - half_col_buffer)] = 1
            right_up_margin = np.zeros((row, col))
            right_up_margin[row - right_cut:, 0:(col - half_col_buffer)] = 1
            right_down_margin = np.zeros((row, col))
            right_down_margin[row - right_cut:, half_col_buffer:col] = 1
            for i in range(row_num - 1, row_num * col_num - 1, row_num):
                weight_array[i] = right_margin
                weight_array[i - 1] = right_border

            if (down_cut == (col - col_buffer)):
                for i in range(row_num * col_num - row_num, row_num * col_num - 1):
                    weight_array[i] = down_border
                weight_array[0] = up_left_border
                weight_array[row_num - 2] = up_right_border
                weight_array[row_num - 1] = right_up_margin
                weight_array[row_num * col_num - row_num - 1] = right_down_margin
                weight_array[row_num * (col_num - 1)] = down_left_border
                weight_array[row_num * col_num - 2] = down_right_border
            else:
                down_margin = np.zeros((row, col))
                down_margin[half_row_buffer:(row - half_row_buffer), col - down_cut:] = 1
                left_down_margin = np.zeros((row, col))
                left_down_margin[0:(row - half_row_buffer), col - down_cut:] = 1
                right_down_margin_2 = np.zeros((row, col))
                right_down_margin_2[half_row_buffer:(row), col - down_cut:] = 1
                for i in range(row_num * col_num - row_num, row_num * col_num - 1):
                    weight_array[i] = down_margin
                    weight_array[i - row_num] = down_border
                weight_array[0] = up_left_border
                weight_array[row_num - 2] = up_right_border
                weight_array[row_num - 1] = right_up_margin
                weight_array[row_num * col_num - row_num - 1] = right_down_margin
                weight_array[row_num * (col_num - 2)] = down_left_border
                weight_array[row_num * (col_num - 1)] = left_down_margin
                weight_array[row_num * col_num - row_num - 2] = down_right_border
                corner_margin = np.zeros((row, col))
                corner_margin[row - right_cut:, col - down_cut:] = 1
                weight_array[row_num * col_num - 2] = right_down_margin_2
                weight_array[row_num * col_num - 1] = corner_margin
                print(corner_margin)
        weight_array = np.array(weight_array)

    return weight_array



## output the one channel data in the same format
def output_same(data, template_file_name, output_name, gdal_type):
    gtif = gdal.Open(template_file_name)
    ## get the first band in the file
    band = gtif.GetRasterBand(1)
    ## get the rows and cols of the input file
    rows = gtif.RasterYSize
    cols = gtif.RasterXSize
    output_format = output_name.split('.')[-1].upper()
    if (output_format == 'TIF'):
        output_format = 'GTIFF'
    elif (output_format == 'RST'):
        output_format = 'rst'
    driver = gdal.GetDriverByName(output_format)
    outDs = driver.Create(output_name, cols, rows, 1, gdal_type)
    outBand = outDs.GetRasterBand(1)
    outBand.WriteArray(data)
    # georeference the image and set the projection
    outDs.SetGeoTransform(gtif.GetGeoTransform())
    outDs.SetProjection(gtif.GetProjection())
    outDs.FlushCache()
    outBand.SetNoDataValue(-99)
    ## need to release the driver
    del outDs
    return output_name


## set the GPU number you want to use
os.environ["CUDA_VISIBLE_DEVICES"]= "1"
project_name = 'unet1'
res_folder = '/home/vishal/DL_Results'
project_folder = os.path.join(res_folder,project_name)
batch_size = 16
if(not os.path.isdir(project_folder)):
    os.mkdir(project_folder)

bands, label = read_scenes(['/workspace/_libs/pond_extensive'])


import random
from sklearn.model_selection import train_test_split
type_list = ['andhra pradesh semi intensive', 'ecuador extensive', 'integrated mangrove', 'intensive', 'long lot extensive', 'smallholder extensive']
source_folder = '/workspace/_libs/pond_final_subclass'
x_train=[]
y_train=[]
x_val=[]
y_val=[]
for i in os.listdir(source_folder):
    if(i in type_list):
        bands, label = read_scenes([os.path.join(source_folder, i)])
        type_size = bands.shape[0]
        random_list = random.sample(range(type_size), 300)
        sample_bands = bands[random_list,:,:,:]
        sample_label = label[random_list,:,:]
        x_train_type, x_val_type, y_train_type, y_val_type = train_test_split(sample_bands, sample_label, test_size=0.2)
        x_train.append(x_train_type)
        x_val.append(x_val_type)
        y_train.append(y_train_type)
        y_val.append(y_val_type)
x_train = np.concatenate(x_train)
y_train = np.concatenate(y_train)
x_val = np.concatenate(x_val)
y_val = np.concatenate(y_val)
np.random.seed(42)
np.random.shuffle(x_train)
np.random.seed(42)
np.random.shuffle(y_train)
np.random.seed(42)
np.random.shuffle(x_val)
np.random.seed(42)
np.random.shuffle(y_val)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(bands, label, test_size=0.2, random_state=42)


x_train, y_train = clarkAug(x_train, y_train)
x_val, y_val = clarkAug(x_val, y_val)
y_train = np.expand_dims(y_train, axis = -1)
y_val = np.expand_dims(y_val, axis = -1)

x_train =x_train.astype('float32')
y_train =y_train.astype('float32')
x_val = x_val.astype('float32')
y_val = y_val.astype('float32')


model_1 = binary_unet(256,256,6)

# model_1 = FCN2(2, 256, 256, 6)
loss = segmentation_models.losses.BinaryFocalLoss()
# sgd = optimizers.SGD(lr=1E-4, decay=5**(-4), momentum=0.9, nesterov=True)
adam_opt = optimizers.Adam(lr=1E-4)
model_1.compile(loss=loss, # 'weighted_categorical_crossentropy' 
              optimizer = adam_opt,
              metrics=['accuracy',segmentation_models.metrics.IOUScore()])  

hist1 = model_1.fit(x_train,y_train,
                  validation_data=(x_val,y_val),
#                   class_weight = 'balanced',       # 10/20/2020 class_weights
                  batch_size= batch_size , 
                  epochs= 2,verbose=1,
                  )


model_path = os.path.join(project_folder, project_name+".h5")
model_1.save(model_path)

hist_path = os.path.join(project_folder, project_name + "_hist.pkl")

import pickle

with open(hist_path, 'wb') as pickle_file:
    pickle.dump(hist1, pickle_file)