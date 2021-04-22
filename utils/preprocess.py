__author__ = 'yunbo'
__revised__ = 'shuxin'

import numpy as np


def reshape_patch(img_tensor, patch_size):  # 进来是(4 , 20 , 1 , 140 , 140)   转为(4,20,140//4,140//4,4*4*1)变为原来1/16大小，通道数变多
    # img_tensor.shape: (batch_size, seq_length, img_height, img_width, num_channels)   #(4 , 20 , 140 , 140 , 1)
    assert 5 == img_tensor.ndim                     #判断一下输入的维度是不是一个5D张量
    img_tensor = img_tensor.permute(0, 1, 3, 4, 2)  #(4 , 20 , 140 ,140 , 1)
    batch_size = np.shape(img_tensor)[0]
    seq_length = np.shape(img_tensor)[1]
    img_height = np.shape(img_tensor)[2]
    img_width = np.shape(img_tensor)[3]
    num_channels = np.shape(img_tensor)[4]
    # reshape the tensor into: (batch_size, seq_length, height/patch, patch_size, width/patch, patch_size, num_channels) #
    a = np.reshape(img_tensor, [batch_size, seq_length, #(4 , 20 , 140//4 , 4 , 140//4 , 4 ,1)
                                img_height // patch_size, patch_size,
                                img_width // patch_size, patch_size,
                                num_channels])
    # transpose into: (batch_size, seq_length, height/patch, width/patch, patch_size, patch_size, num_channels)
    b = np.transpose(a, [0, 1, 2, 4, 3, 5, 6])  #转为( 4 , 20 , 140//4 , 140//4 , 4 , 4 , 1)
    # reshape into: (batch_size, seq_length, height/patch, width/patch, patch_size*patch_size*num_channels)
    patch_tensor = np.reshape(b, [batch_size, seq_length,
                                  img_height // patch_size,
                                  img_width // patch_size,
                                  patch_size * patch_size * num_channels])  #转为(4 , 20 , 140//4 , 140//4 , 4*4*1)
    return patch_tensor 


def reshape_patch_back(patch_tensor, patch_size):
    # patch_tensor.shape: (batch_size, seq_length, patch_height, patch_width, channels)
    assert 5 == patch_tensor.ndim
    batch_size = np.shape(patch_tensor)[0]
    seq_length = np.shape(patch_tensor)[1]
    patch_height = np.shape(patch_tensor)[2]
    patch_width = np.shape(patch_tensor)[3]
    channels = np.shape(patch_tensor)[4]
    # calculate the img_channels
    img_channels = channels // (patch_size * patch_size)
    # reshape into: (batch_size, seq_length, height/patch, width/patch, patch_size, patch_size, num_channels)
    a = np.reshape(patch_tensor, [batch_size, seq_length,
                                  patch_height, patch_width,
                                  patch_size, patch_size,
                                  img_channels])
    # transpose into: (batch_size, seq_length, height/patch, patch_size, width/patch, patch_size, num_channels)
    b = np.transpose(a, [0, 1, 2, 4, 3, 5, 6])
    # reshape into: (batch_size, seq_length, height, width, num_channels)
    img_tensor = np.reshape(b, [batch_size, seq_length,
                                patch_height * patch_size,
                                patch_width * patch_size,
                                img_channels])
    return img_tensor
