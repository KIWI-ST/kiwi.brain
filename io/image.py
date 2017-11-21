# use PIL library to read image format data
from PIL import Image
import numpy as np  
import os

dir = os.getcwd()

def test_pil():  

    #读取图像  
    im = Image.open(dir+"/dataset/1.png")  
    #显示图像  
    im.show()  
  
    #转换成灰度图像  
    im_gray = im.convert("L")  
    im_gray.show()  
  
    #保存图像  
    im_gray.save("/data/image_gray.jpg")  
  
    return 

test_pil()