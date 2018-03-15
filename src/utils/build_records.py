import os
import tensorflow as tf
from PIL import Image
import numpy as np

cwd = 'D:/Workspace/train/'
cwd_train = 'D:/Workspace/train/train/'
files_train = os.listdir(cwd_train)
files_train.sort(key=lambda x: x[:])
writer_train = tf.python_io.TFRecordWriter(
    os.path.join(cwd, 'train.tfrecords'))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

for j in range(len(files_train)):
     class_path = os.path.join(cwd_train, files_train[j])
     for img_name in os.listdir(class_path):
        img_path = class_path+'/'+img_name
        img = Image.open(img_path).convert('L')
        #强制大小为8x8
        img= img.resize((8,8))
        img_raw = img.tobytes()  # 转换成二进制
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'label': _int64_feature(j),
                'image': _bytes_feature(img_raw)
            }))
        writer_train.write(example.SerializeToString())
writer_train.close()


cwd_eval = 'D:/Workspace/train/eval/'
files_eval = os.listdir(cwd_eval)
files_eval.sort(key=lambda x: x[:])
writer_eval = tf.python_io.TFRecordWriter(
    os.path.join(cwd, 'eval.tfrecords'))

for j in range(len(files_eval)):
     class_path = os.path.join(cwd_eval, files_eval[j])
     for img_name in os.listdir(class_path):
        img_path = class_path+'/'+img_name
        img = Image.open(img_path).convert('L')
        #强制大小为8x8
        img= img.resize((8,8))
        img_raw = img.tobytes()  # 转换成二进制
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'label': _int64_feature(j),
                'image': _bytes_feature(img_raw)
            }))
        writer_eval.write(example.SerializeToString())
writer_eval.close()
