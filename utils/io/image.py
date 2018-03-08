# use PIL library to read image format data
from PIL import Image
import numpy as np  
import os

dir = os.getcwd()

#通过相对路径读取图片
def pick_image(relative_path):
    #计算绝对位置
    loc=dir+relative_path
    #读取img
    im = Image.open(loc)
    #转换成Matrix
    width,height = im.size
    im = im.convert("L")
    raw_data = im.getdata()
    #normal操作
    raw_data = np.matrix(raw_data,dtype = 'float')/255.0
    #np 转换成矩阵
    shape_data = np.reshape(raw_data,(width,height))
    return shape_data

#example
#shaped = pick_image('/dataset/1.png')
#print(shaped)